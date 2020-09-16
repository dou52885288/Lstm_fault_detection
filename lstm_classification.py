
import numpy as np
from activators import SigmoidActivator, TanhActivator, SoftmaxActivator


class LstmLayer():
    def __init__(self, input_width, state_width, output_width,
                 learning_rate, penaltyL2, momentum):
        self.input_width = input_width
        self.state_width = state_width
        self.output_width= output_width
        self.learning_rate = learning_rate
        self.penaltyL2 = penaltyL2
        self.momentum = momentum
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        self.class_activator = SoftmaxActivator()
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf, self.vWfh, self.vWfx, self.vbf =(self.init_weight_mat(0))
        # 输入门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wih, self.Wix, self.bi, self.vWih, self.vWix, self.vbi =(self.init_weight_mat(0))
        # 输出门权重矩阵Wfh, Wfx, 偏置项bf
        self.Woh, self.Wox, self.bo, self.vWoh, self.vWox, self.vbo =(self.init_weight_mat(0))
        # 单元状态权重矩阵Wfh, Wfx, 偏置项bf
        self.Wch, self.Wcx, self.bc, self.vWch, self.vWcx, self.vbc =(self.init_weight_mat(0))
        # 下一层权重Wy, 偏值by
        self.Wy, self.by, self.vWy, self.vby =(self.init_weight_mat(1))


    def init_weight_mat(self,i):
        '''
        初始化权重矩阵
        '''
        if (i<1):
            Wh = np.mat(np.random.uniform(-0.5, 0.5,
            (self.state_width, self.state_width)))/self.state_width
            vWh = np.mat(np.zeros(Wh.shape))
            Wx = np.mat(np.random.uniform(-0.5, 0.5,
            (self.state_width, self.input_width)))/self.input_width
            vWx = np.mat(np.zeros(Wx.shape))
            b = np.mat(np.random.uniform(-0.5,0.5,(self.state_width, 1)))/self.state_width
            vb = np.mat(np.zeros(b.shape))
            return Wh, Wx, b, vWh, vWx, vb
        else:
            Wy = np.mat(np.random.uniform(-0.5, 0.5,
            (self.output_width, self.state_width)))/self.output_width
            vWy = np.mat(np.zeros(Wy.shape))
            by = np.mat(np.random.uniform(-0.5,0.5,(self.output_width, 1)))/self.output_width
            vby = np.mat(np.zeros(by.shape))
            return Wy, by, vWy, vby

    def forward(self, x):
        '''
        根据式1-式6进行前向计算
        '''
        self.x = x
        self.reset_state()
        n = x.shape[0]
        for i in range(n):
            # 遗忘门
            fg = self.calc_gate(x[i], self.Wfx, self.Wfh, 
                self.bf, self.gate_activator,i)
            self.f_list[i] = fg
            # 输入门
            ig = self.calc_gate(x[i], self.Wix, self.Wih,
                self.bi, self.gate_activator,i)
            self.i_list[i] = ig
            # 输出门
            og = self.calc_gate(x[i], self.Wox, self.Woh,
                self.bo, self.gate_activator,i)
            self.o_list[i] = og
            # 即时状态
            ct = self.calc_gate(x[i], self.Wcx, self.Wch,
                self.bc, self.output_activator,i)
            self.ct_list[i] = ct
            # 单元状态
            if i==0:
                c = np.multiply(ig, ct)
            else:
                c = np.multiply(fg, self.c_list[i - 1]) + np.multiply(ig, ct)
            self.c_list[i] = c
            # 输出
            h = np.multiply(og, self.output_activator.forward(c))
            self.h_list[i] = h
            y = self.class_activator.forward(self.Wy * h.T + self.by)
            self.y_list[i] = y.T

    def reset_state(self):
        # 各个时刻的单元状态向量c
        self.c_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        # 各个时刻的输出向量h
        self.h_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        # 各个时刻的遗忘门f
        self.f_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        # 各个时刻的输入门i
        self.i_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        # 各个时刻的输出门o
        self.o_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        # 各个时刻的即时状态c~
        self.ct_list = np.mat(np.zeros((self.x.shape[0], self.state_width)))
        self.y_list = np.mat(np.zeros((self.x.shape[0], self.output_width)))

    def calc_gate(self, x, Wx, Wh, b, activator,i):
        '''
        计算门
        '''
        if i==0:
            h = np.mat(np.zeros((1, self.state_width)))
        else:
            h = self.h_list[i-1] # 上次的LSTM输出
        net = (Wh * h.T + Wx * x.T + b).T
        gate = activator.forward(net)
        return gate


    def backward(self, e):
        '''
        实现LSTM训练算法
        '''
        self.e = e
        self.calc_delta()
        self.calc_gradient()
        self.update()

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.vWfh = self.momentum * self.vWfh - self.learning_rate * \
                (self.Wfh_grad + self.penaltyL2 * self.Wfh)
        self.vWfx = self.momentum * self.vWfx - self.learning_rate * \
                        (self.Wfx_grad + self.penaltyL2 * \
                         np.concatenate((np.mat(np.zeros((self.Wfx.shape[0],1))),self.Wfx[:,1:]),axis=1))
        self.vbf = self.momentum * self.vbf - self.learning_rate * self.bf_grad

        self.vWih = self.momentum * self.vWih - self.learning_rate * \
                        (self.Wih_grad + self.penaltyL2 * self.Wih)
        self.vWix = self.momentum * self.vWix - self.learning_rate * \
                        (self.Wix_grad + self.penaltyL2 * \
                         np.concatenate((np.mat(np.zeros((self.Wix.shape[0],1))),self.Wix[:,1:]),axis=1))
        self.vbi = self.momentum * self.vbi - self.learning_rate * self.bi_grad

        self.vWoh = self.momentum * self.vWoh - self.learning_rate * \
                        (self.Woh_grad + self.penaltyL2 * self.Woh)
        self.vWox = self.momentum * self.vWox - self.learning_rate * \
                        (self.Wox_grad + self.penaltyL2 * \
                         np.concatenate((np.mat(np.zeros((self.Wox.shape[0],1))),self.Wox[:,1:]),axis=1))
        self.vbo = self.momentum * self.vbo - self.learning_rate * self.bo_grad

        self.vWch = self.momentum * self.vWch - self.learning_rate * \
                        (self.Wch_grad + self.penaltyL2 * self.Wch)
        self.vWcx = self.momentum * self.vWcx - self.learning_rate * \
                        (self.Wcx_grad + self.penaltyL2 * \
                         np.concatenate((np.mat(np.zeros((self.Wcx.shape[0],1))),self.Wcx[:,1:]),axis=1))
        self.vbc = self.momentum * self.vbc - self.learning_rate * self.bc_grad

        self.vWy = self.momentum * self.vWy - self.learning_rate * \
                       (self.Wy_grad + self.penaltyL2 * self.Wy)
        self.vby = self.momentum * self.vby - self.learning_rate * self.by_grad


        self.Wfh += self.vWfh
        self.Wfx += self.vWfx
        self.bf  += self.vbf
        self.Wih += self.vWih
        self.Wix += self.vWix
        self.bi  += self.vbi
        self.Woh += self.vWoh
        self.Wox += self.vWox
        self.bo  += self.vbo
        self.Wch += self.vWch
        self.Wcx += self.vWcx
        self.bc  += self.vbc
        self.Wy  += self.vWy
        self.by  += self.vby

    def calc_delta(self):
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta()  # 输出误差项
        self.delta_o_list = self.init_delta()  # 输出门误差项
        self.delta_i_list = self.init_delta()  # 输入门误差项
        self.delta_f_list = self.init_delta()  # 遗忘门误差项
        self.delta_ct_list = self.init_delta() # 即时输出误差项
        self.delta_c_list = self.init_delta()  #state c
        self.delta_h_list[-1] = self.e[-1] * self.Wy
        a = self.output_activator.backward(self.output_activator.forward(self.c_list[-1]))
        self.delta_c_list[-1] = np.multiply(self.delta_h_list[-1], self.o_list[-1], a)
        m = self.e.shape[0]
        for k in range(m-1, 0, -1):
            self.calc_delta_k(k)

    def init_delta(self):
        '''
        初始化误差项
        '''
        delta_list = np.mat(np.zeros((self.e.shape[0],self.state_width)))
        return delta_list

    def calc_delta_k(self, k):
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k-1]
        tan_c = self.output_activator.forward(c)
        delta_h = self.delta_h_list[k]
        delta_c = self.delta_c_list[k]
        delta_y = self.e[k-1]
        # 根据式9计算delta_o
        gate_o = np.multiply(tan_c, self.gate_activator.backward(og))
        delta_o = np.multiply(delta_h, gate_o)
        gate_f = np.multiply(c_prev, self.gate_activator.backward(fg))
        delta_f = np.multiply(delta_c, gate_f)
        gate_i = np.multiply(ct, self.gate_activator.backward(ig))
        delta_i = np.multiply(delta_c, gate_i)
        gate_c = np.multiply(ig, self.output_activator.backward(ct))
        delta_ct = np.multiply(delta_c, gate_c)

        delc = np.multiply(og, self.output_activator.backward(tan_c))
        delta_h_prev = np.multiply(delta_h, (gate_o * self.Woh + \
                np.multiply(gate_i, delc) * self.Wih + \
                np.multiply(gate_f, delc) * self.Wfh + \
                np.multiply(gate_c, delc) * self.Wch)) + delta_y * self.Wy
        delc1 = np.multiply(self.o_list[k-1],self.output_activator.backward(self.output_activator.forward(c_prev)))
        delta_c_prev = np.multiply(delta_c, fg) + np.multiply(delta_h_prev, delc1)
        # 保存全部delta值
        self.delta_h_list[k-1] = delta_h_prev
        self.delta_c_list[k-1] = delta_c_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self):
        # 初始化遗忘门权重梯度矩阵和偏置项
        Wfh_grad, Wfx_grad, bf_grad = (
            self.init_weight_gradient_mat(0))
        # 初始化输入门权重梯度矩阵和偏置项
        Wih_grad, Wix_grad, bi_grad = (
            self.init_weight_gradient_mat(0))
        # 初始化输出门权重梯度矩阵和偏置项
        Woh_grad, Wox_grad, bo_grad = (
            self.init_weight_gradient_mat(0))
        # 初始化单元状态权重梯度矩阵和偏置项
        Wch_grad, Wcx_grad, bc_grad = (
            self.init_weight_gradient_mat(0))
        Wy_grad, by_grad = (self.init_weight_gradient_mat(1))
        m = self.e.shape[0]
        # 计算对上一次输出h的权重梯度
        for t in range(m-1, 0, -1):
            h = self.h_list[t]
            h_prev = self.h_list[t - 1]
            x = self.x[t]
            Wfh_grad += self.delta_f_list[t].T * h_prev
            Wfx_grad += self.delta_f_list[t].T * x
            bf_grad += self.delta_f_list[t].T
            Wih_grad += self.delta_i_list[t].T * h_prev
            Wix_grad += self.delta_i_list[t].T * x
            bi_grad += self.delta_i_list[t].T
            Woh_grad += self.delta_o_list[t].T * h_prev
            Wox_grad += self.delta_o_list[t].T * x
            bo_grad += self.delta_o_list[t].T
            Wch_grad += self.delta_ct_list[t].T * h_prev
            Wcx_grad += self.delta_ct_list[t].T * x
            bc_grad += self.delta_ct_list[t].T
            Wy_grad += self.e[t].T * h
            by_grad += self.e[t].T

        self.Wfh_grad = Wfh_grad/(m-1)
        self.Wfx_grad = Wfx_grad/m
        self.bf_grad = bf_grad/m
        self.Wih_grad = Wih_grad/(m-1)
        self.Wix_grad = Wix_grad/m
        self.bi_grad = bi_grad/m
        self.Woh_grad = Woh_grad/(m-1)
        self.Wox_grad = Wox_grad/m
        self.bo_grad = bo_grad/m
        self.Wch_grad = Wch_grad/(m-1)
        self.Wcx_grad = Wcx_grad/m
        self.bc_grad = bc_grad/m
        self.Wy_grad = Wy_grad/m
        self.by_grad = by_grad/m

    def init_weight_gradient_mat(self,i):
        '''
        初始化权重矩阵
        '''
        if i<1:
            Wh_grad = np.mat(np.zeros((self.state_width, self.state_width)))
            Wx_grad = np.mat(np.zeros((self.state_width, self.input_width)))
            b_grad = np.mat(np.zeros((self.state_width, 1)))
            return Wh_grad, Wx_grad, b_grad
        else:
            Wy_grad = np.mat(np.zeros((self.output_width, self.state_width)))
            by_grad = np.mat(np.zeros((self.output_width, 1)))
            return Wy_grad, by_grad





