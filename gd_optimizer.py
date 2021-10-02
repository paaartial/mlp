class gradient_descent_optimizer():
    def __init__(self, a=0.001, b1=0.9, b2=0.999, e=10**-8):
        self.alpha=a
        self.beta1=b1
        self.beta2=b2
        self.epsilon=e
    def adapt_lr(self):
        pass
    def adapt_gc(self):
        pass

class Momentum(gradient_descent_optimizer):
    def __init__(self, b):
        super().__init__(self, b1=b)

    def adapt_gc(self, m_prev, cur_grad):
        # w' = w - aM
        # M = Bmt-1+ (1-B)dL/dw
        return self.beta1*m_prev + (1-self.beta1)*cur_grad

class Adadelta(gradient_descent_optimizer): 
    pass

class Adam(gradient_descent_optimizer):
    pass