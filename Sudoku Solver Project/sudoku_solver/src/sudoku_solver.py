"""
sudoku_solver.py
────────────────
Step 7 of the Sudoku Solver pipeline.

Classic backtracking algorithm:
  1. Find next empty cell
  2. Try digits 1–9
  3. If valid → recurse
  4. If none work → backtrack
"""

import numpy as np
from copy import deepcopy


def is_valid_placement(board, row, col, num):
    """
    Return True if placing `num` at (row, col) does not violate
    any Sudoku constraint (row, column, 3×3 box).
    """
    # Row check
    if num in board[row]:
        return False

    # Column check
    if num in [board[r][col] for r in range(9)]:
        return False

    # 3×3 box check
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == num:
                return False

    return True


def validate_board(board):
    """
    Check that the recognised board has no immediate conflicts.
    Useful to catch CNN misrecognitions before attempting to solve.

    Returns (is_valid: bool, errors: list of (row, col, value))
    """
    errors = []
    b = deepcopy(board) if isinstance(board, list) else board.tolist()

    for r in range(9):
        for c in range(9):
            v = b[r][c]
            if v == 0:
                continue
            b[r][c] = 0
            if not is_valid_placement(b, r, c, v):
                errors.append((r, c, v))
            b[r][c] = v

    return len(errors) == 0, errors


def solve_sudoku(board):
    """
    Backtracking solver.  Modifies `board` (list-of-lists) in place.
    Returns True if a solution was found, False otherwise.
    """
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for num in range(1, 10):
                    if is_valid_placement(board, r, c, num):
                        board[r][c] = num
                        if solve_sudoku(board):
                            return True
                        board[r][c] = 0   # backtrack
                return False              # no digit worked
    return True  # no empty cells → solved


def verify_solution(solved_board):
    """
    Verify that every row, column, and 3×3 box contains exactly 1–9.
    Works with both list-of-lists and numpy arrays.
    """
    arr    = np.array(solved_board)
    target = set(range(1, 10))

    for i in range(9):
        if set(arr[i].tolist())   != target: return False  # row
        if set(arr[:, i].tolist()) != target: return False  # col

    for br in range(3):
        for bc in range(3):
            box = arr[br*3:br*3+3, bc*3:bc*3+3]
            if set(box.flatten().tolist()) != target:
                return False

    return True


def solve(recognised_board):
    """
    High-level entry point:
      1. Validate recognised board
      2. Solve with backtracking
      3. Verify solution

    Args:
        recognised_board : 9×9 list-of-lists or numpy array (0 = empty)

    Returns:
        solved_board  : 9×9 list-of-lists (None if failed)
        stats         : dict with keys is_valid, solved, verified, errors
    """
    board_list = (recognised_board.tolist()
                  if isinstance(recognised_board, np.ndarray)
                  else deepcopy(recognised_board))

    is_valid, errors = validate_board(board_list)
    solved_board     = deepcopy(board_list)
    success          = solve_sudoku(solved_board)
    verified         = success and verify_solution(solved_board)

    return solved_board if verified else None, {
        "is_valid": is_valid,
        "errors"  : errors,
        "solved"  : success,
        "verified": verified,
    }
