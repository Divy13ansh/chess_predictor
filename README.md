# Chess Human Move Predictor

A **learning-to-rank model** that predicts which chess move a 2200-2600 ELO human player is most likely to make, trained on Lichess Elite games. This is the **core project** — the web application is just a wrapper around the model.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why This Approach?](#2-why-this-approach)
3. [Dataset](#3-dataset)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference](#7-inference)
8. [Evaluation](#8-evaluation)
9. [Notebook Structure](#9-notebook-structure)
10. [Web Application](#10-web-application)
11. [Files](#11-files)
12. [Next Steps](#12-next-steps)

---

## 1. Problem Statement

Given a chess position and a list of all legal moves, **rank the moves by how likely a 2200-2600 ELO human player is to play each one**.

This is different from:
- **Chess engines** (Stockfish, AlphaZero): These search for the *best* move by evaluating positions many moves ahead
- **Move prediction**: This learns what humans *actually* play, capturing patterns like favorite openings, tactical mistakes, and positional preferences

### Formal Definition

For a position `P` with legal moves `M = {m1, m2, ..., mk}`, output a scoring function `s(m)` such that higher scores indicate higher probability of being played:

```
ranked = sorted(M, key=lambda m: s(m), reverse=True)
```

The correct move from the game should ideally be ranked #1 (Top-1) or in the top 3 (Top-3).

---

## 2. Why This Approach?

### 2.1 Learning-to-Rank

We treat this as a **ranking problem**, not a classification problem. Here's why:

**Classification approach** (what we tried first):
- Train a binary classifier: is this the played move? (1) or not (0)
- Each move is independent — the model doesn't know about other candidates
- We can get probabilities, but the model doesn't optimize for relative ordering

**Ranking approach** (what we use):
- All moves in a position form a "query group"
- The model sees all candidates together
- Optimizes for **NDCG** (Normalized Discounted Cumulative Gain) — a ranking metric
- Better suited for "which is best among these options"

### 2.2 Why LightGBM LambdaRank?

| Model | Pros | Cons |
|-------|------|------|
| Logistic Regression | Fast, interpretable | Linear — can't capture interactions |
| SGD Classifier | Scales to big data | Same as LR |
| XGBoost | Good gradient boosting | Slower than LightGBM |
| Neural Networks | Can learn complex patterns | Needs more data, harder to tune |
| **LightGBM Ranker** | Fast, handles sparse, LambdaRank objective | Needs group data |

LightGBM is ideal because:
- **Sparse features**: 802 features where most are 0 (empty squares)
- **Many negatives**: 15-29 candidates per position
- **Speed**: Tree-based models are fast for inference
- **LambdaRank**: Directly optimizes ranking metrics

### 2.3 Dataset Construction Strategy

For each position in each game:

```
Position: board state before a move
Played move: the move actually made (label = 1)
Legal moves: all other moves that could have been made (label = 0)
```

This creates a **labeled dataset** where we know the "correct" answer (the played move) and "incorrect" answers (all other legal moves).

---

## 3. Dataset

### 3.1 Source Data

```bash
# Download from Lichess
# https://lichess.org/db
# File: lichess_elite_2019-10.pgn
```

** Lichess Elite** refers to games played by high-rated players. The October 2019 export is commonly used because:
- Large volume (~50M games)
- Good quality games with accurate ELO ratings
- Clean PGN format with game metadata

### 3.2 Filtering Criteria

```python
# Filter for ELO 2200-2600 (both players)
min_elo = 2200
max_elo = 2600

for game in pgn_file:
    w_elo = int(game.headers["WhiteElo"])
    b_elo = int(game.headers["BlackElo"])
    
    if min_elo <= w_elo <= max_elo and min_elo <= b_elo <= max_elo:
        games.append(game)
```

Why this range?
- **2200+**: Players who understand positional concepts, opening theory
- **2600**: Grandmaster level (reasonable sample size)
- Both players in range ensures balanced game quality

### 3.3 Statistics

| Metric | Value |
|--------|-------|
| Total games loaded | 15,000 |
| ELO range | 2200-2600 |
| Average game length | ~60 moves |
| Time control | Blitz (180+0) |

### 3.4 Game-Level Train/Test Split

**Critical**: We split at the **game level**, not position level.

```python
from sklearn.model_selection import train_test_split

train_games, test_games = train_test_split(
    games, test_size=0.2, random_state=42
)
```

**Why?** If we split by positions:
- Same game could have positions in both train and test
- The model could "memorize" patterns from a specific game
- This is **data leakage** — would inflate accuracy artificially

By splitting games:
- All positions from a game go entirely to train OR test
- Test positions are truly unseen
- This gives **honest** evaluation

---

## 4. Feature Engineering

We create **802 features** for each (position, move) pair.

### 4.1 Board Encoding (768 features)

The board has 64 squares, and each can contain one of 12 piece types (6 pieces × 2 colors). We use **one-hot encoding**:

```
[white pawn at square 0] [white knight at square 0] ... [black king at square 63]
  (feature 0)              (feature 1)                  (feature 767)
```

```python
def board_to_array(board):
    """
    Convert chess board to 768-dimensional feature vector.
    
    Layout (768 features):
    - Features 0-5:   white pawn, knight, bishop, rook, queen, king at square 0
    - Features 6-11:  same for square 1
    - ...
    - Features 762-767: same for square 63
    
    Example: If there's a white knight at square 10:
    - First find feature index: 10 * 12 + 1 (knight is index 1)
    - Set that feature to 1, all others to 0
    """
    arr = [0] * 768
    
    for piece_type in chess.PIECE_TYPES:
        # White pieces (indices 0-5)
        for square in board.pieces(piece_type, chess.WHITE):
            idx = piece_type - 1  # pawn=1 -> index 0
            arr[square * 12 + idx] = 1
        
        # Black pieces (indices 6-11)
        for square in board.pieces(piece_type, chess.BLACK):
            idx = piece_type - 1 + 6  # offset by 6
            arr[square * 12 + idx] = 1
    
    return arr
```

**Why one-hot?**
- Binary (0 or 1) — easy for trees to split on
- No arbitrary ordering between pieces
- 768 dimensions capture full board state

### 4.2 Move Semantic Features (24 features)

These describe the move itself in a semantically meaningful way:

| Feature | Type | Description |
|---------|-----|-------------|
| moving_pawn | binary | Is moving piece a pawn? |
| moving_knight | binary | Is moving piece a knight? |
| moving_bishop | binary | Is moving piece a bishop? |
| moving_rook | binary | Is moving piece a rook? |
| moving_queen | binary | Is moving piece a queen? |
| moving_king | binary | Is moving piece a king? |
| moving_is_white | binary | Is moving piece white? |
| moving_is_black | binary | Is moving piece black? |
| from_rank | int (0-7) | From square rank (0=white side) |
| from_file | int (0-7) | From square file (0=a-file) |
| to_rank | int (0-7) | To square rank |
| to_file | int (0-7) | To square file |
| rank_distance | int (0-7) | Absolute rank change |
| file_distance | int (0-7) | Absolute file change |
| is_capture | binary | Does move capture something? |
| captured_pawn | binary | Captured piece type |
| captured_knight | binary | ... |
| captured_bishop | binary | ... |
| captured_rook | binary | ... |
| captured_queen | binary | ... |
| captured_king | binary | ... |
| gives_check | binary | Does move give check? |
| is_promotion | binary | Is this a promotion? |
| is_castling | binary | Is this castling? |

```python
def move_to_semantic_array(board, move):
    """
    Convert a move to 24 semantic features.
    """
    arr = []
    piece = board.piece_at(move.from_square)
    
    # Moving piece type (6 binary)
    piece_type_onehot = [0] * 6
    if piece is not None:
        piece_type_onehot[piece.piece_type - 1] = 1
    arr.extend(piece_type_onehot)
    
    # Moving piece color (2 binary)
    if piece is not None:
        arr.append(1 if piece.color == chess.WHITE else 0)
        arr.append(1 if piece.color == chess.BLACK else 0)
    else:
        arr.extend([0, 0])
    
    # From/to coordinates (4 integers)
    from_rank = move.from_square // 8
    from_file = move.from_square % 8
    to_rank = move.to_square // 8
    to_file = move.to_square % 8
    arr.extend([from_rank, from_file, to_rank, to_file])
    
    # Distance (2 integers)
    arr.extend([
        abs(to_rank - from_rank),
        abs(to_file - from_file)
    ])
    
    # Capture info (7 features)
    captured_piece = board.piece_at(move.to_square)
    is_capture = 1 if captured_piece is not None else 0
    arr.append(is_capture)
    
    captured_type_onehot = [0] * 6
    if captured_piece is not None:
        captured_type_onehot[captured_piece.piece_type - 1] = 1
    arr.extend(captured_type_onehot)
    
    # Check / promotion / castling (3 binary)
    board_copy = board.copy()
    board_copy.push(move)
    arr.extend([
        1 if board_copy.is_check() else 0,
        1 if move.promotion is not None else 0,
        1 if board.is_castling(move) else 0
    ])
    
    return arr  # 24 values
```

**Why not just use from/to square IDs?**
- Old approach: 128 one-hot columns for from_square and to_square
- Problem: Treated like "from square 4 is halfway between 3 and 5" — fake ordinal relationship
- Solution: One-hot piece types avoid this, use raw coordinates only where arithmetic is valid (distance)

### 4.3 Position Metadata Features (10 features)

These describe the overall game state:

| Feature | Type | Description |
|---------|-----|-------------|
| w_k_castle | binary | White can kingside castle? |
| w_q_castle | binary | White can queenside castle? |
| b_k_castle | binary | Black can kingside castle? |
| b_q_castle | binary | Black can queenside castle? |
| black_to_move | binary | Is it Black's turn? |
| en_passant | binary | Is en passant available? |
| mobility | int | Number of legal moves available |
| white_material | int | Sum of white piece values |
| black_material | int | Sum of black piece values |
| material_diff | int | white_material - black_material |

```python
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def get_metadata_features_fast(board):
    """Fast metadata features — no ELO needed."""
    piece_map = board.piece_map().values()
    wm = sum(PIECE_VALUES[p.piece_type] for p in piece_map if p.color == chess.WHITE)
    bm = sum(PIECE_VALUES[p.piece_type] for p in piece_map if p.color == chess.BLACK)
    
    return [
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        int(board.turn == chess.BLACK),
        int(board.ep_square is not None),
        board.legal_moves.count(),
        wm, bm, wm - bm
    ]
```

### 4.4 Feature Analysis Summary

| Feature Set | Dimensions | Description |
|-------------|------------|-------------|
| Board | 768 | One-hot piece positions |
| Move | 24 | Move semantics |
| Metadata | 10 | Game state |
| **Total** | **802** | |

After removing dead features:
- **Dead features**: 40 (always 0 or always same value)
- **Live features**: 762 (used for training)

---

## 5. Model Architecture

### 5.1 LightGBM LambdaRank

The final model is a **LightGBM Ranker** with LambdaRank objective:

```python
import lightgbm as lgb

ranker = lgb.LGBMRanker(
    objective="lambdarank",     # Ranking objective
    metric="ndcg",              # Optimize NDCG
    n_estimators=400,         # Number of trees
    num_leaves=128,             # Leaves per tree
    learning_rate=0.03,         # Step size
    feature_fraction=0.8,      # Column sampling
    bagging_fraction=0.8,     # Row sampling
    bagging_freq=1,             # Bagging frequency
    min_data_in_leaf=20,        # Regularization
    verbosity=-1,
    n_jobs=-1,
)
```

### 5.2 How LambdaRank Works

LambdaRank is a **listwise** ranking algorithm. Here's the intuition:

1. **Each position is a "query"**: All legal moves in a position form one group
2. **Labels**: Played move = 1 (relevant), other moves = 0 (not relevant)
3. **Lambda computation**: For each candidate, compute a "lambda" (importance weight) based on how much swapping it with a higher-ranked candidate would improve NDCG
4. **Gradient boosting**: Trees are trained to predict these lambda values

```python
# Simplified intuition
def compute_lambda(scores, labels):
    """
    For each candidate, compute how much it matters.
    If a low-scoring candidate is ranked above a high-scoring one,
    that's a big error -> high lambda.
    """
    lambdas = [0] * len(scores)
    # ... compute NDCG gradient for each pair ...
    return lambdas
```

### 5.3 Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators` | 400 | More trees = more capacity, risk overfit |
| `num_leaves` | 128 | Leaves per tree — controls complexity |
| `learning_rate` | 0.03 | Lower = more trees needed, better generalization |
| `feature_fraction` | 0.8 | Use 80% of features per tree — regularization |
| `bagging_fraction` | 0.8 | Use 80% of samples per tree — regularization |
| `min_data_in_leaf` | 20 | Minimum samples in leaf — prevents overfitting |

---

## 6. Training Pipeline

### 6.1 Dataset Creation

Two datasets are built:

**df_ranker** (for ranker training):
- 1 positive + 15 negatives per position
- More negatives = better ranking contrast
- Expected shape: ~6M rows

**df_dataset_balanced** (for classifiers):
- 1 positive + 3 negatives per position
- Balanced dataset
- Expected shape: ~3M rows

```python
def create_dataset_parallel(games, num_negatives=15, min_move_number=0):
    """
    For each position:
    - Get the played move (label=1)
    - Sample num_negatives unplayed moves (label=0)
    - Extract features for each (board, move) pair
    """
    rows = []
    
    for game_idx, game in enumerate(games):
        board = game.board()
        
        for move_idx, move in enumerate(game.mainline_moves()):
            if move_idx < min_move_number:
                board.push(move)
                continue
            
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            
            # Positive: played move
            played_move = move
            board_arr = board_to_array(board)
            move_arr = move_to_semantic_array(board, played_move)
            meta_arr = get_metadata_features_fast(board)
            
            rows.append(
                board_arr + move_arr + meta_arr + [1, game_idx, move_idx]
            )
            
            # Negatives: unplayed legal moves
            negatives = random.sample(
                [m for m in legal_moves if m != played_move],
                min(num_negatives, len(legal_moves) - 1)
            )
            
            for neg in negatives:
                move_arr = move_to_semantic_array(board, neg)
                rows.append(
                    board_arr + move_arr + meta_arr + [0, game_idx, move_idx]
                )
            
            board.push(move)
    
    return pd.DataFrame(rows)
```

### 6.2 Preparing Group Data

LightGBM Ranker requires **group sizes** — how many candidates are in each group:

```python
def build_groups(sorted_group_ids):
    """
    Convert sorted game IDs to group sizes.
    
    Example: game_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    Output:      [3, 2, 4]
    """
    groups = []
    current_id = sorted_group_ids[0]
    count = 0
    
    for gid in sorted_group_ids:
        if gid == current_id:
            count += 1
        else:
            groups.append(count)
            current_id = gid
            count = 1
    
    groups.append(count)
    return groups

# Sort by game_id for LightGBM
train_order = np.argsort(game_ids_train)
X_train = X_train[train_order]
y_train = y_train[train_order]
groups_train = build_groups(game_ids_train[train_order])
```

### 6.3 Training

```python
ranker.fit(
    X_train,           # Feature matrix (n_samples, 762)
    y_train,          # Labels (n_samples,)
    group=groups_train  # Group sizes (n_queries,)
)
```

### 6.4 Why GroupShuffleSplit?

We use `GroupShuffleSplit` to create train/validation splits **within training games**:

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_ranker, y_ranker, groups=game_ids_ranker))
```

This ensures:
- All positions from one game go to train OR validation
- No position from test_games leaks into training
- The overfit test compares train games vs test games

---

## 7. Inference

### 7.1 How Inference Works

At inference time, given a position:

```python
def predict_best_move_with_scores(board):
    """
    1. Get all legal moves
    2. For each move, extract features (same as training)
    3. Run model.predict() to get scores
    4. Sort by score descending
    """
    legal_moves = list(board.legal_moves)
    
    # Build feature matrix
    feature_matrix = []
    for move in legal_moves:
        features = board_to_array(board) + \
                   move_to_semantic_array(board, move) + \
                   get_metadata_features_fast(board)
        feature_matrix.append(features)
    
    X = np.array(feature_matrix)
    
    # Get scores from model
    scores = ranker.predict(X)
    
    # Rank moves by score
    ranked = sorted(zip(legal_moves, scores), key=lambda x: -x[1])
    
    return ranked  # [(move, score), ...] sorted by score
```

### 7.2 Feature Extraction at Inference

The key insight: **features are extracted BEFORE the move is pushed**, using the current board state plus the candidate move's semantics:

```python
# Current board state
board_feats = board_to_array(board)           # 768 features

# Move-specific features (what this move WOULD do)
move_feats = move_to_semantic_array(board, move)  # 24 features

# Game state at this moment
meta_feats = get_metadata_features_fast(board)    # 10 features

# Full feature vector = 768 + 24 + 10 = 802
# We drop the 40 "dead" features, using 762
```

### 7.3 Why This Works

The model learns patterns like:
- "In this board configuration, knights on b1 and f1 favor e3 over d4"
- "When there's an enemy queen on the same file, rook moves are more likely"
- "In positions with 2 bishops vs. queen, exchanges are popular"

These are move preferences conditioned on board positions — exactly what we want.

---

## 8. Evaluation

### 8.1 Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Top-1** | Correct move ranked #1 | `correct ranked #1 / total` |
| **Top-3** | Correct move in top 3 | `correct in top 3 / total` |
| **Gap** | Train-Test difference | `train_top1 - test_top1` |

### 8.2 Evaluation Function

```python
def evaluate_top1_full_ranker(games, model, max_positions=2000):
    """
    Evaluate Top-1 accuracy on held-out games.
    
    For each position in test games:
    1. Generate features for all legal moves
    2. Get model scores
    3. Check if played move is ranked #1
    """
    correct = 0
    total = 0
    
    for game in games:
        board = game.board()
        
        for actual_move in game.mainline_moves():
            if total >= max_positions:
                return correct / total
            
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            
            # Build feature matrix
            rows = []
            for m in legal_moves:
                row = board_to_array(board) + \
                      move_to_semantic_array(board, m) + \
                      get_metadata_features_fast(board)
                rows.append(row)
            
            # Predict scores
            X = csr_matrix(rows)
            scores = model.predict(X)
            
            # Best predicted move
            best_idx = np.argmax(scores)
            best_move = legal_moves[best_idx]
            
            # Check correctness
            if best_move == actual_move:
                correct += 1
            
            total += 1
            board.push(actual_move)
    
    return correct / total
```

### 8.3 Typical Results

From the notebook (on held-out test games):

| Model | Train Top-1 | Test Top-1 | Gap | Test Top-3 |
|-------|-------------|------------|-----|------------|
| Logistic Regression | 0.25 | 0.22 | 0.03 | 0.38 |
| LightGBM (binary) | 0.31 | 0.25 | 0.06 | 0.42 |
| **LightGBM Ranker** | **0.34** | **0.29** | 0.05 | **0.48** |

### 8.4 Interpreting Results

- **~30% Top-1**: The model correctly picks the played move 30% of the time
- **~48% Top-3**: The played move is in the top 3 predictions nearly half the time
- **Gap < 0.05**: Acceptable — some overfitting but not severe

Is 30% good? Consider:
- Random would be ~1/20 = 5% (average 20 legal moves)
- 30% is 6× better than random
- Human experts might be ~50-60% (they also make "mistakes")

---

## 9. Notebook Structure

The notebook `chess_human_move_predictor_v4.ipynb` has 13 sections:

| # | Section | Lines | Description |
|---|---------|-------|-------------|
| 1 | **Setup** | ~100 | Imports, config, load PGN |
| 2 | **Game EDA** | ~200 | ELO distribution, openings |
| 3 | **Position Extraction** | ~200 | Extract all positions from games |
| 4 | **Move EDA** | ~150 | Move frequencies, legal counts |
| 5 | **Position Features EDA** | ~150 | Material, castling, phase |
| 6 | **Feature Engineering** | ~350 | Feature extraction functions |
| 7 | **Feature Analysis** | ~350 | Audit zero/constant columns |
| 8 | **Dataset Creation** | ~450 | Build labeled datasets |
| 9 | **Dataset EDA** | ~300 | Validate label balance |
| 10 | **Model Training** | ~2000 | All models with evaluation |
| 11 | **Model Comparison** | ~500 | Side-by-side comparison |
| 12 | **Save Model** | ~50 | Pickle serialization |
| 13 | **Summary** | ~50 | What was built, next steps |

---

## 10. Web Application

The **Flask web app** wraps the model for interactive use.

### 10.1 Architecture

```
┌─────────────────���─���─────────────────────┐
│          Browser (index.html)            │
│   Chess UI with drag/drop moves         │
└──────────────┬──────────────────────────┘
               │ HTTP POST /api/*
               ▼
┌─────────────────────────────────────────┐
│          Flask (app.py)                 │
│  - Load pickled model at startup       │
│  - /api/best-move: rank moves           │
│  - /api/apply-move: make move           │
└──────────────┬──────────────────────────┘
               │ pickle.load()
               ▼
┌─────────────────────────────────────────┐
│   lgbm_ranker_optimized.pkl             │
│   LightGBM Ranker model                 │
└─────────────────────────────────────────┘
```

### 10.2 API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | — | Render HTML |
| `/api/best-move` | POST | `{"fen": "..."}` | Best move + top 5 ranked |
| `/api/legal-moves` | POST | `{"fen": "..."}` | List all legal moves |
| `/api/apply-move` | POST | `{"fen": "...", "uci": "..."}` | New FEN after move |
| `/api/status` | GET | — | Is model loaded? |

### 10.3 Example Usage

```python
import requests

fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3"

response = requests.post(
    "http://localhost:5000/api/best-move",
    json={"fen": fen}
)
data = response.json()

print(f"Best move: {data['best_move']}")
print(f"Top 5:")
for move in data['top5']:
    print(f"  {move['rank']}. {move['san']} ({move['score']:.4f})")
```

---

## 11. Files

```
chess_predictor/
├── README.md                         # This file
├── app.py                          # Flask API server
├── templates/
│   └── index.html                 # Interactive chess UI
├── chess_human_move_predictor_v4.ipynb  # Training notebook
├── lgbm_ranker_optimized.pkl       # Trained model
├── requirements.txt               # Dependencies
└── .gitignore                     # Ignore .pkl, .pgn, __pycache__
```

**Required but not included**:
- `lichess_elite_2019-10.pgn` — Download from Lichess database

---

## 12. Next Steps

### 12.1 Immediate Improvements

1. **More training data**: Currently uses 5,000 games → scale to 12,000+
2. **Early stopping**: Add validation set to find optimal `n_estimators`
3. **More negatives**: Increase from 15 to 25-29 (full legal set)
4. **Hyperparameter tuning**: Grid search on `learning_rate`, `num_leaves`

### 12.2 New Features

1. **Attacked squares**: Which squares does this move attack?
2. **Defended squares**: Which squares does this move defend?
3. **Center control**: Is the move to/between center squares?
4. **Mobility after**: How many moves available after this move?
5. **King safety**: Distance to enemy king

### 12.3 Advanced Approaches

1. **Neural networks**: Transformer or CNN on board state
2. **ELO conditioning**: Separate models per ELO band
3. **Opening book**: Learn opening preferences specifically
4. **Temporal patterns**: Last N moves as context
5. **Multi-task**: Predict ELO + move together

---

## References

- [Lichess Database](https://lichess.org/db) — Download PGN games
- [LightGBM Ranking](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#ranking) — LambdaRank docs
- [python-chess](https://python-chess.readthedocs.io/) — Chess library
- [Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank) — Overview of ranking algorithms
- [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) — Ranking metric