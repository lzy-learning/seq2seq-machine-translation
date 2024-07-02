class AverageMeter(object):
    '''
    统计平均值，可以是时间或者损失函数值的平均值
    '''

    def __init__(self):
        self.reset()
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

