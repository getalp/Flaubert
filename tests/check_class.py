
class A(object):
    num = 2
    def __init__(self):
        self.num = 5

class B(A):
    #num = 4
    def __init__(self):
        super().__init__()
        self.num = 10   

def main():
    a = A()
    b = B()
    print('a.num =', a.num)
    print('b.Num =', b.num)
    print('A.num =', A.num)
    print('B.Num =', B.num)

if __name__ == "__main__":
    main()