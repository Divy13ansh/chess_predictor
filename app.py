import pickle
import chess
import chess.pgn
import numpy as np
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load your model ──────────────────────────────────────────────────────────
MODEL_PATH = "lgbm_ranker_optimized_copy.pkl"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[✓] Model loaded from {MODEL_PATH}")
else:
    print(f"[!] No model found at '{MODEL_PATH}'. "
          f"Set MODEL_PATH env var or place lgbm_ranker.pkl next to app.py")


# ── Feature extraction ───────────────────────────────────────────────────────
def board_to_features(board: chess.Board, move: chess.Move):
    """768 board features + 34 move/game features = 802 total"""

    piece_map = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,
        (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10,
        (chess.KING,   chess.BLACK): 11,
    }

    # ── 768 one-hot board ──────────────────────────────────────────
    board_feats = [0.0] * (12 * 64)
    for square, piece in board.piece_map().items():
        idx = piece_map[(piece.piece_type, piece.color)]
        board_feats[idx * 64 + square] = 1.0

    # ── Moving piece info ──────────────────────────────────────────
    moving_piece = board.piece_at(move.from_square)
    pt    = moving_piece.piece_type if moving_piece else None
    color = moving_piece.color      if moving_piece else None

    moving_pawn     = int(pt == chess.PAWN)
    moving_knight   = int(pt == chess.KNIGHT)
    moving_bishop   = int(pt == chess.BISHOP)
    moving_rook     = int(pt == chess.ROOK)
    moving_queen    = int(pt == chess.QUEEN)
    moving_king     = int(pt == chess.KING)
    moving_is_white = int(color == chess.WHITE) if color is not None else 0
    moving_is_black = int(color == chess.BLACK) if color is not None else 0

    # ── Square geometry ────────────────────────────────────────────
    from_rank     = chess.square_rank(move.from_square)
    from_file     = chess.square_file(move.from_square)
    to_rank       = chess.square_rank(move.to_square)
    to_file       = chess.square_file(move.to_square)
    rank_distance = abs(to_rank - from_rank)
    file_distance = abs(to_file - from_file)

    # ── Capture info ───────────────────────────────────────────────
    captured = board.piece_at(move.to_square)
    if not captured and pt == chess.PAWN and move.to_square == board.ep_square:
        captured_pt = chess.PAWN   # en-passant
    else:
        captured_pt = captured.piece_type if captured else None

    is_capture      = int(captured_pt is not None)
    captured_pawn   = int(captured_pt == chess.PAWN)
    captured_knight = int(captured_pt == chess.KNIGHT)
    captured_bishop = int(captured_pt == chess.BISHOP)
    captured_rook   = int(captured_pt == chess.ROOK)
    captured_queen  = int(captured_pt == chess.QUEEN)
    captured_king   = int(captured_pt == chess.KING)

    # ── Check / mobility (push then pop) ──────────────────────────
    board.push(move)
    gives_check = int(board.is_check())
    mobility    = board.legal_moves.count()
    board.pop()

    is_promotion = int(move.promotion is not None)

    # ── s_castling (was "is_castling" in your feature list) ───────
    s_castling = int(board.is_castling(move))

    # ── Castling rights (before the move) ─────────────────────────
    w_k_castle = int(board.has_kingside_castling_rights(chess.WHITE))
    w_q_castle = int(board.has_queenside_castling_rights(chess.WHITE))
    b_k_castle = int(board.has_kingside_castling_rights(chess.BLACK))
    b_q_castle = int(board.has_queenside_castling_rights(chess.BLACK))

    # ── Side / en passant ──────────────────────────────────────────
    black_to_move = int(board.turn == chess.BLACK)
    en_passant    = int(board.ep_square is not None)

    # ── Material ───────────────────────────────────────────────────
    PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN:  9, chess.KING:   0,
    }
    white_material = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values() if p.color == chess.WHITE
    )
    black_material = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values() if p.color == chess.BLACK
    )
    material_diff = white_material - black_material

    # ── Assemble in EXACT training column order ────────────────────
    # Matches df.drop(["label","position_id","game_id"]) column order:
    # square_0_piece_0 ... square_63_piece_11  (768)
    # then the 34 below:
    extra_feats = [
        moving_pawn, moving_knight, moving_bishop, moving_rook,
        moving_queen, moving_king,
        moving_is_white, moving_is_black,
        from_rank, from_file, to_rank, to_file,
        rank_distance, file_distance,
        is_capture,
        captured_pawn, captured_knight, captured_bishop,
        captured_rook, captured_queen, captured_king,
        gives_check, is_promotion,
        s_castling,                          # ← was is_castling
        w_k_castle, w_q_castle, b_k_castle, b_q_castle,
        black_to_move, en_passant,
        mobility,
        white_material, black_material, material_diff,
        # position_id and game_id are NOT included (dropped before training)
    ]

    assert len(extra_feats) == 34, f"Expected 34 extra features, got {len(extra_feats)}"
    assert len(board_feats) + len(extra_feats) == 802, "Total must be 802"

    return board_feats + extra_feats

def predict_best_move_with_scores(board: chess.Board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, []

    if model is None:
        move = legal_moves[0]
        return move, [{"move": move, "uci": move.uci(), "san": board.san(move), "score": 0.0}]

    # ── Build feature matrix — pass each move BEFORE pushing it ───
    feature_matrix = []
    for move in legal_moves:
        feature_matrix.append(board_to_features(board, move))   # <-- pass move

    try:
        X = np.array(feature_matrix)
        print(f"[DEBUG] Feature matrix shape: {X.shape}")   # should be (N, 803)
        raw_scores = model.predict(X)

        print("\n" + "="*60)
        print(f"  FEN : {board.fen()}")
        print(f"  Moves evaluated : {len(legal_moves)}")
        print(f"  Score range     : min={raw_scores.min():.6f}  max={raw_scores.max():.6f}")
        print(f"  Unique values   : {len(np.unique(raw_scores))}")
        print("-"*60)

        paired = sorted(zip(legal_moves, raw_scores), key=lambda x: float(x[1]), reverse=True)

        print("  RANK  SAN        UCI       SCORE")
        for i, (mv, sc) in enumerate(paired):
            san = board.san(mv)
            print(f"  {i+1:>3}.  {san:<10} {mv.uci():<8}  {float(sc):.6f}{' ◄' if i==0 else ''}")
        print("="*60 + "\n")

        scored = [
            {"move": mv, "uci": mv.uci(), "san": board.san(mv), "score": float(sc)}
            for mv, sc in paired
        ]

    except Exception as e:
        print(f"[!] Model prediction error: {e}")
        import traceback; traceback.print_exc()
        move = legal_moves[0]
        return move, [{"move": move, "uci": move.uci(), "san": board.san(move), "score": 0.0}]

    return scored[0]["move"], scored

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/best-move", methods=["POST"])
def best_move():
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    if board.is_game_over():
        return jsonify({"is_game_over": True, "result": board.result(), "legal_moves": []})

    best_move, scored = predict_best_move_with_scores(board)

    if best_move is None:
        return jsonify({"error": "No legal moves available"}), 400

    top5 = [
        {
            "rank":  i + 1,
            "uci":   m["uci"],
            "san":   m["san"],
            "score": round(m["score"], 6),   # 6 decimals so tiny diffs show
        }
        for i, m in enumerate(scored[:5])
    ]

    san = board.san(best_move)

    return jsonify({
        "best_move":    best_move.uci(),
        "uci":          best_move.uci(),
        "san":          san,
        "from_sq":      chess.square_name(best_move.from_square),
        "to_sq":        chess.square_name(best_move.to_square),
        "legal_moves":  [m.uci() for m in board.legal_moves],
        "is_game_over": False,
        "model_loaded": model is not None,
        "top5":         top5,
    })


@app.route("/api/legal-moves", methods=["POST"])
def legal_moves():
    data = request.get_json(force=True)
    fen = data.get("fen", chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    moves = []
    for m in board.legal_moves:
        moves.append({
            "uci":     m.uci(),
            "from_sq": chess.square_name(m.from_square),
            "to_sq":   chess.square_name(m.to_square),
            "san":     board.san(m),
        })
    return jsonify({"legal_moves": moves, "fen": board.fen()})


@app.route("/api/apply-move", methods=["POST"])
def apply_move():
    data = request.get_json(force=True)
    fen  = data.get("fen", chess.STARTING_FEN)
    uci  = data.get("uci", "")

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify({"error": f"Invalid UCI move: {uci}"}), 400

    if move not in board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    san = board.san(move)
    board.push(move)
    is_over = board.is_game_over()
    return jsonify({
        "new_fen":      board.fen(),
        "san":          san,
        "uci":          uci,    
        "is_game_over": is_over,
        "result":       board.result() if is_over else None,
    })


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)