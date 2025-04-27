class Board:

    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.turn = 'X'
        self.winner = None
        self.moves = 0
        self.game_over = False
    
    def reset(self):
        """Reset the board to start a new game."""
        self.__init__()
    
    def get_state(self):
        """Hashable representation for RL (tuple of tuples)."""
        return tuple(tuple(row) for row in self.board)
    
    def print_board(self):
        print("Current board:")
        for row in self.board:
            print("-" * 21)
            for cell in row:
                print(f"| {cell} ", end="")
            print("|")
        print("-" * 21)
        print(f"Current turn: {self.turn}")
        print(f"Moves made: {self.moves}")
    
    def make_move(self, row, col):
        if self.board[row][col] != ' ':
            print("Cannot move here")
            return
        self.board[row][col] = self.turn
        self.moves += 1

        if self.turn == 'X':
            self.turn = 'O'
        else:
            self.turn = 'X'
        
        state = self.check_state()
        if state != 'C':
            self.winner = state
            self.game_over = True

    def legal_moves(self):
        """List of (row, col) tuples for every empty cell."""
        return [(r, c)
            for r in range(3)
            for c in range(3)
            if self.board[r][c] == ' ']
        
    def check_state(self):
        """
        Check for a winner or draw.
        Returns:
            'X' if X wins,
            'O' if O wins,
            'D' if draw,
            'C' if the game should continue.
        Also sets self.winner and self.game_over.
        """
        lines = []

        for i in range(3):
            lines.append(self.board[i])                   
            lines.append([self.board[r][i] for r in range(3)])  

        
        lines.append([self.board[i][i] for i in range(3)])
        lines.append([self.board[i][2 - i] for i in range(3)])

        
        for line in lines:
            if line[0] != ' ' and line.count(line[0]) == 3:
                self.winner = line[0]
                self.game_over = True
                return self.winner

        
        if self.moves >= 9:
            self.winner = 'D'
            self.game_over = True
            return 'D'

        return 'C'

    