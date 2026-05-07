import pygame
import struct
import time
import json
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

class JoyStick:
    #按键定义
    LaxiX = 0.0    #左摇杆X轴，axis[0]
    LaxiY = 0.0    #左摇杆Y轴，axis[1]
    RaxiX = 0.0    #右摇杆X轴，axis[2]
    RaxiY = 0.0    #右摇杆Y轴，axis[3]
    hatX = 0       #方向键X轴，hat[0]
    hatY = 0       #方向键Y轴，hat[1]
    butA = 0       #A键，but[0]
    butB = 0       #B键，but[1]
    butX = 0       #X键，but[3]
    butY = 0       #Y键，but[4]
    L1 = 0         #L1键，but[6]
    R1 = 0         #R1键，but[7]
    L2 = 0         #L2键，but[8]
    R2 = 0         #R2键，but[9]
    SELECT = 0     #SELECT键，but[10]
    START = 0      #START键，but[11]
    
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        
    def zero_states(self):
        self.LaxiX = 0.0
        self.LaxiY = 0.0
        self.RaxiX = 0.0
        self.RaxiY = 0.0
        self.hatX = 0
        self.hatY = 0
        self.butA = 0
        self.butB = 0
        self.butX = 0
        self.butY = 0
        self.L1 = 0
        self.R1 = 0
        self.L2 = 0
        self.R2 = 0
        self.SELECT = 0
        self.START = 0

    def initjoystick(self):
        # ----- [原版代码] -----
        # pygame.init()
        # pygame.joystick.init()
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # ---------------------

        # ----- [修改代码: 增加防崩溃和错误捕获] -----
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
            else:
                self.joystick = None
        except:
            self.joystick = None
        # ---------------------

    def getjoystickstates(self):
        # ----- [原版代码] -----
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()
        # 
        # for event in pygame.event.get():  # User did something
        #     if event.type == pygame.JOYAXISMOTION:
        #         self.LaxiX = self.joystick.get_axis(0)
        #         self.LaxiY = self.joystick.get_axis(1)
        #         ... (按键映射事件循环) ...
        # ---------------------

        # ----- [修改代码: 直接拉取状态，提高性能，加入异常捕获防闪退] -----
        try:
            pygame.event.pump() # 强制刷新底层硬件状态队列
            
            # 如果手柄未连接（开机忘了开），尝试重新识别
            if self.joystick is None:
                pygame.joystick.quit()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                else:
                    self.zero_states()
                    return

            # 极速强制拉取物理状态（放弃迟缓的 event.get 事件循环机制）
            self.LaxiX = self.joystick.get_axis(0)
            self.LaxiY = self.joystick.get_axis(1)
            self.RaxiX = self.joystick.get_axis(2)
            self.RaxiY = self.joystick.get_axis(3)
            hat = self.joystick.get_hat(0)
            self.hatX = hat[0]
            self.hatY = hat[1]
            self.butA = self.joystick.get_button(0)
            self.butB = self.joystick.get_button(1)
            self.butX = self.joystick.get_button(3)
            self.butY = self.joystick.get_button(4)
            self.L1 = self.joystick.get_button(6)
            self.R1 = self.joystick.get_button(7)
            self.L2 = self.joystick.get_button(8)
            self.R2 = self.joystick.get_button(9)
            self.SELECT = self.joystick.get_button(10)
            self.START = self.joystick.get_button(11)
            
            pygame.event.clear() # 清理事件缓存，防止内存泄漏
            
        except Exception:
            # 若出现设备断开、USB读取错误，立刻清零执行紧急制动
            self.joystick = None
            self.zero_states()
        # ---------------------
        
    def display(self):
        print('================')
        print('Axies:')
        print('LaxiX: {}'.format(self.LaxiX))
        print('LaxiY: {}'.format(self.LaxiY))
        print('RaxiX: {}'.format(self.RaxiX))
        print('RaxiY: {}'.format(self.RaxiY))
        print('----------------')
        print('Hat:')
        print('hatX: {}'.format(self.hatX))
        print('hatY: {}'.format(self.hatY))
        print('----------------')
        print('button:')
        print('butA: {}'.format(self.butA))
        print('butB: {}'.format(self.butB))
        print('butX: {}'.format(self.butX))
        print('butY: {}'.format(self.butY))
        print('L1: {}'.format(self.L1))
        print('R1: {}'.format(self.R1))
        print('L2: {}'.format(self.L2))
        print('R2: {}'.format(self.R2))
        print('SELECT: {}'.format(self.SELECT))
        print('START: {}'.format(self.START))
        print('================')
                
joy = JoyStick()

def init_joystick():
    global joy
    joy.initjoystick()

def read_joystick():
    global joy
    joy.getjoystickstates()
    
    result = {
        "LaxiX": 0.,
        "LaxiY": 0.,
        "RaxiX": 0.,
        "RaxiY": 0.,
        "hatX": 0,
        "hatY": 0,
        "butA": 0,
        "butB": 0,
        "butX": 0,
        "butY": 0,
        "L1": 0,
        "R1": 0,
        "L2": 0,
        "R2": 0,
        "SELECT": 0,
        "START": 0,
    }

    result["LaxiX"] = joy.LaxiX
    result["LaxiY"] = joy.LaxiY
    result["RaxiX"] = joy.RaxiX
    result["RaxiY"] = joy.RaxiY
    
    result["hatX"] = joy.hatX
    result["hatY"] = joy.hatY
    
    result["butA"] = joy.butA
    result["butB"] = joy.butB
    result["butX"] = joy.butX
    result["butY"] = joy.butY
    
    result["L1"] = joy.L1
    result["R1"] = joy.R1
    result["L2"] = joy.L2
    result["R2"] = joy.R2
    
    result["SELECT"] = joy.SELECT
    result["START"] = joy.START
    
    return json.dumps(result)