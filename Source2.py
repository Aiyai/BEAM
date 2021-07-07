import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from numpy.core.arrayprint import format_float_positional

class BEAM:

    def __init__(self, sample_signals, for_test=False):
        #상태 표시에 사용할 신호의 개수를 정합니다. 신호는 a+bi의 형태로 저장되기 때문에 상태는 (samples, 2)의 행렬로서 저장되게 됩니다.
        self.sample_signals = sample_signals
        
        #기록 여부를 결정한다.
        self.for_test = for_test

        self.LAMBDA = 0.001
        self.DELTA = 0.5
        self.N_ANTENNA = 16

        #송신기와 수신기 사이의 거리 d와 각도 t_angle의 초기값을 설정한다.
        self.d = 20
        self.t_angle = 90
        self.speed = 2

        #시작 시 총 보상과 현재 보상, 게임 수를 초기화한다.
        self.total_reward = 0
        self.current_reward = 0.
        self.total_game = 0

        self.channel = np.ones((16,1)) + np.random.rand(16,1) * 0.02

        #초기 빔 각도를 설정한다. 이 때, 첫 각도를 알고있다고 가정하기 때문에 t_angle의 초기값과 같아진다.
        #여기서 BEAM_ANGLE1은 reward를 받기 위한 빔포머이며, BEAM_ANGLE2는 위상을 알기 위한 빔포머 값이다. 각도가 범위를 넘으면 허수부값은 -가 된다.
        self.BEAM_ANGLE = 90

        #상태에 들어갈 신호 집합 행렬을 만듭니다.
        #신호가 A + jB의 형태로 나타날 때, 이를 [A, B]로서 저장합니다.
        #신호의 개수가 N개라고 할 때 이 행렬은 (N, 2)의 shape를 가집니다.
        self.signal_before_action = np.zeros((sample_signals, 2))
        self.signal_after_action = np.zeros((sample_signals, 2))

        if for_test:
            self.OBJ = np.zeros((2, self.movecount + 1))
            self.BEM = np.zeros(self.movecount + 1)


    def _get_state(self):
        
        for i in range(self.sample_signals):
            SIGNAL = self.beamformer(self.N_ANTENNA, self.channel, self.DELTA, self.LAMBDA, self.BEAM_ANGLE + i, self.d, self.t_angle)
            self.signal_before_action[i, 0] = SIGNAL.real
            self.signal_before_action[i, 1] = SIGNAL.imag
        
        STATE = np.concatenate((self.signal_before_action, self.signal_after_action), axis=0)

        return STATE


    def reset(self):
        self.current_reward = 0
        self.total_game = 0

        self.d = 20
        self.t_angle = 90
        
        self.BEAM_ANGLE = 90
        self.movecount = 1000
        self.channel = np.ones((16,1)) + np.random.rand(16,1) * 0.02
        
        for i in range(self.sample_signals):
            SIGNAL = self.beamformer(self.N_ANTENNA, self.channel, self.DELTA, self.LAMBDA, self.BEAM_ANGLE + i, self.d, self.t_angle)
            self.signal_before_action[i, 0] = SIGNAL.real
            self.signal_before_action[i, 1] = SIGNAL.imag

        return self._get_state()
        


    def _update_BEAM(self, move):  #move = 1(move beam clockwise) , 0(stop), -1(move beam counter-clockwise)
        #Moves Beam angle with given move
        self.BEAM_ANGLE = self.BEAM_ANGLE + move

        #Resets BEAM_ANGLE within range [0, 180]
        if self.BEAM_ANGLE >= 180:
            self.BEAM_ANGLE = self.BEAM_ANGLE - 180
        elif self.BEAM_ANGLE <= 0:
            self.BEAM_ANGLE = self.BEAM_ANGLE + 180
        
        for i in range(self.sample_signals):
            SIGNAL = self.beamformer(self.N_ANTENNA, self.channel, self.DELTA, self.LAMBDA, self.BEAM_ANGLE + i, self.d, self.t_angle)
            self.signal_after_action[i, 0] = SIGNAL.real
            self.signal_after_action[i, 1] = SIGNAL.imag

        reward = (self.signal_after_action[0, 0]**2 + self.signal_after_action[0, 1]**2)**0.5 - (self.signal_before_action[0, 0]**2 + self.signal_before_action[0, 1]**2)**0.5

        if self.for_test:
            self.BEM[self.movecount] = self.BEAM_ANGLE

        return reward


    def _update_channel(self):
        self.channel = self.channel + np.random.rand(16,1) * 0.01
    
    def _is_gameover(self):

        self.movecount = self.movecount - 1

        if self.movecount == 0:
            self.total_reward += self.current_reward

            return True
        else:
            return False

    def _update_transmitter(self, speed):
        
        #Transmitter moves randomly with given speed
        self.d = self.d + speed * (np.random.rand(1) - 0.5)
        self.t_angle = self.t_angle + speed * (np.random.rand(1) + 0.5)

        if self.for_test:
            self.OBJ[0, self.movecount] = self.d
            self.OBJ[1, self.movecount] = self.t_angle


    def step(self, action):
        # action: 0: 반시계, 1: 유지, 2: 시계
        state = self._get_state()

        Tracking_reward = self._update_BEAM(action - 1)
        stable_reward = 0.00 if action == 1 else 0
        
        gameover = self._is_gameover()
        #
        self._update_transmitter(self.speed)
        self._update_channel()

        if gameover:
            pass
        else:
            self.reward = Tracking_reward + stable_reward
            self.current_reward += self.reward

        return state, self.reward, gameover
            

    def beamformer(self, N_ANTENNA, channel, DELTA, LAMBDA, BEAM_ANGLE, d, t_angle):
            SUM = 0
            for i in range(N_ANTENNA):
                SUM = SUM + channel[i] * 0.25 * exp(-2j * 3.141592 * d / LAMBDA) * exp(-2j * 3.141592 * i * DELTA * np.cos(np.deg2rad(BEAM_ANGLE))) * exp(2j * 3.141592 * i * DELTA * np.cos(np.deg2rad(t_angle)))
            return SUM

    def testing(self):
        return self.OBJ, self.BEM

