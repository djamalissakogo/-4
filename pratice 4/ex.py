import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Среда для игры в крестики-нолики
class TicTacToe:
    def __init__(self):
        # Инициализация пустой доски 3x3
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Агент всегда первый и играет "X"

    def reset(self):
        # Сброс доски к начальному состоянию
        self.board.fill(0)
        self.current_player = 1
        return self.board

    def step(self, action):
        # Выполнение действия, обновление доски и проверка окончания игры
        if self.board[action] != 0:
            raise ValueError("Invalid action!")
        self.board[action] = self.current_player
        reward, done = self.check_game_over()
        self.current_player *= -1  # Смена игрока
        return self.board, reward, done

    def check_game_over(self):
        # Проверка условий победы/проигрыша или ничьи
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return 1 if self.current_player == 1 else -1, True
        if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return 1 if self.current_player == 1 else -1, True
        return (0, True) if not (self.board == 0).any() else (0, False)

    def available_actions(self):
        # Возвращение доступных действий
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

# Простой противник
class SimpleAgent:
    def __init__(self, strategy="random"):
        self.strategy = strategy  # Стратегия противника

    def choose_action(self, env):
        actions = env.available_actions()
        if self.strategy == "random":
            return random.choice(actions)  # Случайный выбор действия
        center, corners = (1, 1), [(0, 0), (0, 2), (2, 0), (2, 2)]
        return center if center in actions else random.choice([c for c in corners if c in actions] or actions)

# Агент Q-learning с измененной жадной стратегией и обновлением таблицы значений
class QLearningAgent:
    def __init__(self, alpha=0.02, gamma=0.9, exploration_prob=0.05):
        # Таблица значений, инициализирующая состояния нормальных значений как 0.5, победные состояния как 1, и проигрышные состояния как 0
        self.q_table = defaultdict(lambda: np.full((3, 3), 0.5))
        self.alpha = alpha  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.exploration_prob = exploration_prob  # Фиксированная вероятность исследования 0.05

    def get_state(self, board):
        # Преобразование доски в кортеж для использования в Q-таблице
        return tuple(map(tuple, board))

    def choose_action(self, state, available_actions):
        # Выбор действия с жадной стратегией с фиксированной вероятностью исследования
        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(available_actions)  # Исследование
        return max(available_actions, key=lambda x: self.q_table[state][x])  # Эксплуатация

    def update_q_value(self, state, action, reward, next_state, done):
        # Обновление значения Q (Update Q-value)
        current_q_value = self.q_table[state][action]
        next_max_q_value = np.max(self.q_table[next_state]) if not done else (1 if reward == 1 else 0 if reward == -1 else 0.5)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_max_q_value - current_q_value)
        self.q_table[state][action] = new_q_value

    def learn(self, env, opponent, episodes=10000):
        rewards = []  # Список для хранения вознаграждений
        for episode in range(episodes):
            state, total_reward, done = self.get_state(env.reset()), 0, False

            while not done:
                if env.current_player == 1:  # Агент всегда начинает первым
                    available_actions = env.available_actions()
                    action = self.choose_action(state, available_actions)
                    next_board, reward, done = env.step(action)
                    next_state = self.get_state(next_board)
                    self.update_q_value(state, action, reward, next_state, done)
                    state, total_reward = next_state, total_reward + reward
                else:
                    env.step(opponent.choose_action(env))
            rewards.append(total_reward)

        return rewards

# Функция скользящего среднего для сглаживания графика
def moving_average(data, window_size=2000):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Инициализация среды и противника
env, opponent = TicTacToe(), SimpleAgent("random")

# Обучение агента и построение графика среднего вознаграждения
agent = QLearningAgent(alpha=0.02, gamma=0.9, exploration_prob=0.05)
rewards = agent.learn(env, opponent)

# Расчет среднего вознаграждения
average_reward = np.mean(rewards)
print("Average Reward:", average_reward)

# Расчет скользящего среднего вознаграждения и построение графика
smoothed_rewards = moving_average(rewards, window_size=2000)
plt.plot(smoothed_rewards)
plt.xlabel("Эпизоды (Episodes)")
plt.ylabel("Среднее вознаграждение (Average Reward)")
plt.title("Прогресс обучения агента (Agent Learning Progress)")

plt.show()
