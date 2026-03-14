"""
Sudoku solving using backtracking algorithm
"""

import numpy as np


def is_valid(board, row, col, num):
    """
    Check if placing num at board[row][col] is valid
    
    Args:
        board: 9x9 sudoku board (0 for empty)
        row, col: Position to check
        num: Number to place (1-9)
        
    Returns:
        True if valid, False otherwise
    """
    # Check row
    if num in board[row]:
        return False
    
    # Check column
    if num in board[:, col]:
        return False
    
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    box = board[box_row:box_row+3, box_col:box_col+3]
    if num in box:
        return False
    
    return True


def solve_sudoku(board):
    """
    Solve sudoku using backtracking
    
    Args:
        board: 9x9 numpy array (0 for empty cells)
        
    Returns:
        True if solved, False if no solution
    """
    # Find next empty cell
    empty = np.where(board == 0)
    
    if len(empty[0]) == 0:
        return True  # Solved!
    
    row, col = empty[0][0], empty[1][0]
    
    # Try numbers 1-9
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            
            if solve_sudoku(board):
                return True
            
            # Backtrack
            board[row, col] = 0
    
    return False


if __name__ == "__main__":
    # Test with simple sudoku
    test_board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    print("Test sudoku:")
    print(test_board)
    
    if solve_sudoku(test_board):
        print("\nSolved!")
        print(test_board)
    else:
        print("\nNo solution found")
