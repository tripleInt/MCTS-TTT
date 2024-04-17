from __future__ import annotations  # Python 3.7 need(PEP563)
import numpy as np
from typing import List
from random import shuffle
import time


# ============================================================
#                 实现 各种游戏相关基本操作
# ============================================================


class Chess:
    def __init__(self, name: str, val: int) -> None:
        self.name: str = name
        self.val: int = val

    def __repr__(self) -> str:
        return f"Chess Object({str(self.name)}, {str(self.val)})"


# 操作
class Move:
    def __init__(self, x: int, y: int, chess: Chess) -> None:
        self.x: int = x
        self.y: int = y
        self.chess: Chess = chess

    def __repr__(self) -> str:
        return (
            "Move Object[("
            + str(self.x)
            + ", "
            + str(self.y)
            + ") , "
            + str(self.chess)
            + "]"
        )


STATUS = {0: " ", 1: "O", -1: "X"}
X = Chess(STATUS[-1], -1)
O = Chess(STATUS[1], 1)


# 状态
class State:
    def __init__(self, nxtMove, checkerboardStat: np.array, winNeed: int = -1) -> None:
        """
        Args:
                nxtMove: 接下来该谁下棋了
                checkerboardStat (2 D 网格棋盘):
                    棋盘状态
                winNeed (int, optional):
                    连续多少个棋子可获得胜利. Defaults to -1.
        """
        if len(checkerboardStat.shape) != 2:
            raise Exception("checkerboardStat must be 2D array")

        if checkerboardStat.shape[0] != checkerboardStat.shape[1]:
            raise Exception("checkerboardStat must be square")

        self.checkerboard: np.array = checkerboardStat
        if winNeed == -1:
            winNeed = self.checkerboard.shape[0]
        self.winNeed = winNeed
        self.nxtMove: Chess = nxtMove

    @property
    def checkerboardSize(self):
        return self.checkerboard.shape[0]

    @property
    def result(self):
        """判断游戏结果

        Returns:
                Chess | 0 | None: 若返回 0 代表游戏平局
                            返回 None 表示游戏还未结束，
                            否则返回 X/O(Chess 对象) 表示赢家。
        """
        # 横纵连续
        for i in range(self.checkerboardSize - self.winNeed + 1):
            xSum = np.sum(self.checkerboard[i : i + self.winNeed, :], axis=0)
            ySum = np.sum(self.checkerboard[:, i : i + self.winNeed], axis=1)

            if xSum.min() == -self.winNeed or ySum.min() == -self.winNeed:
                return X
            if xSum.max() == self.winNeed or ySum.max() == self.winNeed:
                return O

        # 对角线连续
        for i in range(self.checkerboardSize - self.winNeed + 1):
            for j in range(self.checkerboardSize - self.winNeed + 1):
                subCheckerboard = self.checkerboard[
                    i : i + self.winNeed, j : j + self.winNeed
                ]
                # 两条斜向对角线
                diag1Sum, diag2Sum = (
                    subCheckerboard.trace(),
                    np.fliplr(subCheckerboard).trace(),
                )

                if diag1Sum == -self.winNeed or diag2Sum == -self.winNeed:
                    return X
                if diag1Sum == self.winNeed or diag2Sum == self.winNeed:
                    return O

        if np.all(self.checkerboard != 0):
            # 平局
            return 0

        # 游戏还未结束
        return None

    @property
    def isOver(self) -> bool:
        """游戏是否结束

        Returns:
                bool: 游戏是否结束
        """
        return self.result is not None

    def getMoves(self) -> List[Move]:
        """获取所有可能的走法

        Returns:
                List[Move]
        """
        return [
            Move(d[0], d[1], self.nxtMove)
            for d in list(zip(*np.where(self.checkerboard == 0)))
        ]

    def couldMove(self, move: Move) -> bool:
        """判断走法是否合法

        Args:
                move (Move)

        Returns:
                bool: 是否合法
        """
        if move.chess != self.nxtMove:
            # 下棋的一方不对
            return False

        if not (
            0 <= move.x < self.checkerboardSize and 0 <= move.y < self.checkerboardSize
        ):
            # 位置不合法
            return False

        # 这位置还没有棋子
        return self.checkerboard[move.x, move.y] == 0

    # def doMove(self, move):
    def doMove(self, move: Move) -> State:  # Python 3.7 need(PEP 563)
        if not self.couldMove(move):
            raise Exception("Move must be legel")

        newCheckerboard = self.checkerboard.copy()
        newCheckerboard[move.x, move.y] = move.chess.val

        if self.nxtMove == X:
            nxtMove = O
        elif self.nxtMove == O:
            nxtMove = X

        # return type(self)(nxtMove, newCheckerboard, self.winNeed)
        return State(nxtMove, newCheckerboard, self.winNeed)  # Python 3.7 need(PEP 563)

    def show(self, outputFn: function = print) -> None:
        """显示当前棋盘状态

        Args:
                outputFn (function, optional):
                    输出的函数. Defaults to print.
        """

        board = np.copy(self.checkerboard)

        def strLines(r):
            return " " + " | ".join(map(lambda x: STATUS.get(int(x), " "), r)) + " "

        for r in board[:-1]:
            outputFn(strLines(r))
            outputFn("-" * (len(r) * 4 - 1))

        outputFn(strLines(board[-1]))
        outputFn()


# ============================================================
#                 实现 蒙特卡洛树搜索
# ============================================================


# 结点
class MCTSNode:
    def __init__(self, stat, fa=None):
        """
        Args:
                stat (State): 结点对应状态
                fa (MCTSNode, optional): 父结点. Defaults to None.
        """
        self.stat: State = stat
        self.fa: MCTSNode = fa

        # 子结点列表
        self.sons: List[MCTSNode] = []

        self._visits = 0  # 已访问过结点
        self._results = {}
        self._notTried = None

    @property
    def isFullyExpanded(self):
        return len(self.notTried) == 0

    @property
    def notTried(self):
        if self._notTried is None:
            self._notTried = self.stat.getMoves()

            # 通过打乱实现“随机”
            shuffle(self._notTried)
        return self._notTried

    # 利用 UCB 算法
    def bestSon(self, c=1.5):
        return self.sons[
            np.argmax(
                [
                    (nod.q / nod.n) + c * np.sqrt((2 * np.log(self.n)) / nod.n)
                    for nod in self.sons
                ]
            )
        ]

    @property
    def q(self):
        v = self.fa.stat.nxtMove.val
        return self._results.get(v, 0) - self._results.get(-1 * v, 0)

    @property
    def n(self):
        return self._visits

    @property
    def isEnd(self) -> bool:
        """是否是终端结点（叶子结点）"""
        return self.stat.isOver

    def expand(self) -> MCTSNode:
        stat = self.stat.doMove(self.notTried.pop())

        son: MCTSNode = MCTSNode(stat, self)
        self.sons.append(son)
        return son

    def rollout(self):
        stat: State = self.stat
        while not stat.isOver:
            stat = stat.doMove(np.random.choice(stat.getMoves()))
        return stat.result

    def backpropagate(self, result):
        self._visits += 1
        self._results[result] = self._results.get(result, 0) + 1
        if self.fa is not None:
            self.fa.backpropagate(result)


# 搜索树
class MCTS(object):
    def __init__(self, rootNod: MCTSNode):
        """蒙特卡洛树

        Args:
                rootNod (MCTSNode): 根结点
        """
        self.rootNod: MCTSNode = rootNod

    def chooseNod(self) -> MCTSNode:
        """选择要扩展的结点

        Returns:
                MCTSNode
        """

        cur = self.rootNod

        # 递归到叶子结点并返回
        while not cur.isEnd:
            if not cur.isFullyExpanded:
                return cur.expand()
            else:
                cur = cur.bestSon()
        return cur

    def bstAction(self, simulationTimes: int = None, duration: float = None):
        """根据 UCB 算法进行搜索扩展，找到最佳操作

        Args:
                simulationTimes (int, optional):
                    为找到最佳操作已经模拟的次数. Defaults to None.
                duration (float, optional):
                    算法搜索的时间（秒）. Defaults to None.
        """

        if simulationTimes is None:
            if duration is None:
                raise Exception("duration must be set")
            endTime: float = time.time() + duration
            while time.time() <= endTime:
                nod = self.chooseNod()
                nod.backpropagate(nod.rollout())
        else:
            for _ in range(simulationTimes):
                nod = self.chooseNod()
                nod.backpropagate(nod.rollout())

        # 展开
        return self.rootNod.bestSon(c=0.0)


# ============================================================
#                 实现 程序主体
# ============================================================
def main():
    # 棋盘
    board_size = 7
    checkerboardStat = np.zeros((board_size, board_size), dtype=int)
    # 游戏
    game = State(X, checkerboardStat, 4)
    # 蒙特卡洛树搜索
    while game.result is None:
        game.show()
        mcts = MCTS(MCTSNode(game))
        bstNod = mcts.bstAction(simulationTimes=2)
        game = bstNod.stat

    result = game.result
    if type(result) == Chess:
        print("Game Over! Winner is:" + STATUS.get(result.val, "Unknown"))
    else:
        print("Game Over! Tie!")

    print("End At: ")
    game.show()


if __name__ == "__main__":
    main()
