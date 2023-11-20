import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class VehicularHoneypotEnv(gym.Env):

    def __init__(self, render: bool = False, seed = None):
        """
        Initializes the Vehicle Honeypot environment.

        Parameters:
        - max_steps (int): The max step
        - initial_risk (float): The initial risk of the IoVs environment.
        - security_risk(float): The current security risk of the IoV environment.

        - initial_resource (int): The initial resource of the autonomous vehicle.
        - residual_resource (int): The residual resource of the autonomous vehicle
        - resource_lower_bound (int): The minimum amount of resources that must be maintained for normal vehicle operation.

        - attacker_launched: 攻击者是否发起了攻击, 初始值为 False, 即不发动攻击
        - attacker_captured: 攻击者是否被捕获
        """
        self.observation_space = spaces.Box(low=np.array([0.0, 0]), high=np.array([1.0, 100]), dtype=np.float32)

        # action space
        self.action_space = spaces.Discrete(4)  # 动作包括不开启蜜罐、低交互蜜罐、中等交互蜜罐、高交互蜜罐

        # 初始化随机种子
        self.seed_value = seed
        self.seed(seed)

        self.security_risk = 0  # 当前车联网安全风险

        # 车辆剩余资源、最大资源限制 和 资源下限
        self.residual_resource = 100    # 车辆剩余资源
        self.resource_lower_bound = 10  # 车辆资源的下限
        self.attack_captured = False    # 攻击者是否被捕获
        self.attack_launched = False    # 攻击者是否发起攻击

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        执行一个 action 并返回观测、奖励、完成标志和信息
        Args:
            action: int, 防御者的 action，其中 0 表示不开启蜜罐，1 表示开启低交互蜜罐，2 表示开启中等交互蜜罐，3 表示开启高交互蜜罐

        Returns:
            observation: np.ndarray, 当前状态信息，包括车联网环境的安全危险程度和车辆的剩余资源
            reward: float, 防御者在这个时间步的奖励值，根据防御者是否成功捕获攻击者而有所不同
            done: bool, 是否完成任务，本环境中 done 始终为 False
            info: dict, 包含一些附加信息，包括攻击者是否被捕获，消耗的资源，奖励等
        """

        # 先判断资源是否满足
        if self.residual_resource < self.resource_lower_bound:
            observation = np.array([self.security_risk, self.residual_resource], dtype=np.float32)
            reward = -5
            done = True
            info = {'residual_resource': self.residual_resource}
            return observation, reward, done, info

        # 判断 action 是否合法
        assert self.action_space.contains(action), f"Action {action} is invalid"
        print("action:", action)

        # 计算这一步车辆的剩余资源
        resource_consumption = action
        self.residual_resource -= resource_consumption  #减去开启相应蜜罐所消耗的资源
        self.residual_resource -= 1 # 车辆行驶所消耗的资源
        print("Current residual resource:", self.residual_resource)
        """
        这里的问题是：如何进行攻击发生的判断：
        方案1：用攻击者发动攻击的概率 和 当前环境安全风险进行对比来确定是否发生攻击
        方案2：根据当前环境的安全风险来随机生成 是否发动攻击。        

        attack_occurred = np.random.choice([True, False], p=[0.1, 0.9])
        上面表示 选择 True 的概率为 0.1，选择 False 的概率为 0.9
        在论文中，defender 的先验概率越大，那么 atacker 发动攻击的概率就小。
        我们可以根据环境的安全风险程度来确定，当安全风险大的时候，攻击者发动攻击概率就小
        因此，我们可以做一个判断,如果危险程度大于 0.5，那么攻击的概率就小一些，
        比如：security_risk = 0.6,那么，就是[0.4,0.6]        
               
        终止条件:
            1. 车辆的资源消耗完
            2. 当 defender 采取 normal system 时，被攻击。也就是 action == 0 以及 attack_launched 
        """

        # 随机均匀地产生当前的 security_risk 值
        self.security_risk = self.np_random.uniform(0.1, 0.9)
        print("security_risk:", self.security_risk)

        # 确定攻击者是否发动攻击
        self.attack_launched = self.np_random.choice([True, False], p=[1 - self.security_risk, self.security_risk])
        print("attack_launched:", self.attack_launched)
        if self.attack_launched:
            if action == 0:
                self.attack_captured = False
                observation = np.array([self.security_risk, self.residual_resource], dtype=np.float32)
                info = {"attacker_launched": self.attack_launched, "attacker_captured": self.attack_captured}
                done = True
                reward = -(10 + self.security_risk * 10)
                return observation, reward, done, info
            else:
                self.attack_captured = True
                reward = action + self.security_risk * 10  # 增加基于安全风险的奖励机制
                observation = np.array([self.security_risk, self.residual_resource], dtype=np.float32)
                info = {"attacker_launched": self.attack_launched, "attacker_captured": self.attack_captured}
                done = False
                return observation, reward, done, info

        else:
            if action == 0:
                reward = 3
            else:
                reward = - action * ()
                reward = -2 * (action + self.security_risk * 2)  # 基于安全风险和资源消耗的奖励
            self.attack_captured = False
            observation = np.array([self.security_risk, self.residual_resource], dtype=np.float32)
            info = {"attacker_launched": self.attack_launched, "attacker_captured": self.attack_captured}
            done = False
            return observation, reward, done, info


    def reset(self):
        """
        reset the vehicular honeypot environment.
        :param:
            security_risk
            residual_resource
            attacker_launched
            honeypot_deployed
            honeypot_interact
        :return: security_risk and residual_resource
        """

        self.security_risk = self.np_random.uniform(0.1, 0.9)
        self.residual_resource = 100
        observation = np.array([self.security_risk, self.residual_resource], dtype=np.float32)
        return observation

    def render(self):
        pass

    def close(self):
        pass

if __name__ == "__main__":

    env = VehicularHoneypotEnv()
    seed = 1
    env.seed(seed)
    # 重置环境以获得初始观察
    observation = env.reset()

    # 定义测试的步数
    num_steps = 10
    for _ in range(num_steps):
        # 随机选择一个动作（示例中使用随机动作，你可以根据你的策略进行选择）
        action = env.action_space.sample()  # 随机选择一个动作
        print(f"Action taken: {action}")

        # 在环境中执行动作并获得新的状态、奖励、完成状态和信息
        observation, reward, done, info = env.step(action)

        # 打印环境的反馈信息
        print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")
        # print(f"Info: {info}")

        # 如果游戏结束，重置环境
        if done:
            observation = env.reset()

    # 关闭环境
    env.close()