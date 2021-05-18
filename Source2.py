import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

class BEAM:

    LAMBDA = 0.001
    DELTA = 0.5 #ANTENNA_DISTANCE / LAMBDA
    N_ANTENNA = 16

    def __init__(self, NOISE_POWER):
        #송신기와 수신기 사이의 거리 d와 각도 t_angle의 초기값을 설정한다.
        self.d = 50
        self.t_angle = 90

        #총 보상과 현재 보상, 게임 수를 초기화한다.
        self.total_reward = 0
        self.current_reward = 0.
        self.total_game = 0
        
        #초기 빔 각도를 설정한다. 이 때, 첫 각도를 알고있다고 가정하기 때문에 t_angle의 초기값과 같아진다.
        #여기서 BEAM_ANGLE1은 reward를 받기 위한 빔포머이며, BEAM_ANGLE2는 위상을 알기 위한 빔포머 값이다. 최댓값에서 각도가 더 가게 되면 허수부값은 -가 된다.
        self.BEAM_ANGLE1 = self.t_angle
        self.BEAM_ANGLE2 = self.t_angle + 30

        #빔포밍되는 
        self.movecount = 200

        self.NOISE_POWER = NOISE_POWER
        self.NOISE = self.NOISE_POWER * np.random.randn(1, 1)

    def _get_state(self):
        pass
                            
        


    def reset(self):
        self.current_reward = 0
        self.total_game = 0

        self.BEAM_ANGLE1 = 0
        self.BEAM_ANGLE2 = 30  

        self.movecount = 200
        self.A_t = [0, 50]

        return self._get_state()
        

    def _update_transmitter(self, speed):
        
        #Transmitter moves randomly with given speed
        self.d = self.d + speed * (np.random(1, 1) - 0.5)
        self.t_angle = self.t_angle + speed * (np.random(1, 1) - 0.5)


    def _update_BEAM(self, move):  #move = 1(move beam clockwise) , 0(stop), -1(move beam counter-clockwise)

        self.PREV_BEAM = self.BEAM_ANGLE1
        #Moves Beam angle with given move
        self.BEAM_ANGLE1 = self.BEAM_ANGLE1 + move * -self.BEAM_ANGLE

        #Resets BEAM_ANGLE within range [0, 180]
        if self.BEAM_ANGLE1 >= 180:
            self.BEAM_ANGLE1 = self.BEAM_ANGLE1 - 180
            self.BEAM_ANGLE2 = self.BEAM_ANGLE1 + 30
        elif self.BEAM_ANGLE1 <= 0:
            self.BEAM_ANGLE1 = self.BEAM_ANGLE1 + 180
        
        SIGNAL1 = self.beamformer(self.N_ANTENNA, self.DELTA, self.BEAM_ANGLE1, self.d, self.t_angle)
        SIGNAL2 = self.beamformer(self.N_ANTENNA, self.DELTA, self.BEAM_ANGLE2, self.d, self.t_angle)

        reward = abs(SIGNAL1)
        return reward


    def _is_gameover(self):

        movecount = self.movecount - 1

        if movecount == 0:
            self.total_reward += self.current_reward

            return True
        else:
            return False


    def step(self, action):
        # action: 0: 반시계, 1: 유지, 2: 시계

        Tracking_reward = self._update_BEAM(action - 1)
        stable_reward = 0.1 if action == 1 else 0
        
        gameover = self._is_gameover()
        self._update_transmitter()

        if gameover:
            pass
        else:
            reward = Tracking_reward + stable_reward
            self.current_reward += reward

        return reward, gameover
            

    def beamformer(N_ANTENNA, DELTA, LAMBDA, BEAM_ANGLE, d, t_angle):
        SUM = 0
        for i in range(N_ANTENNA):
            SUM = SUM + 0.25 * exp(-2j * np.pi * d / LAMBDA) * exp(-2j * np.pi * i * DELTA * np.cos(np.deg2rad(BEAM_ANGLE))) * exp(2j * np.pi * i * DELTA * np.cos(np.deg2rad(t_angle)))
        return SUM

