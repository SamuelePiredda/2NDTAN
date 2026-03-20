class Test:

    __att:int

    pub_att:int

    def __init__(self, testo:str) -> None:
        print(testo)
    
    def fun(self, a:int) -> None:
        self.__att = a
        self.__fun() 

    def __fun(self) -> None:
        print("Variable updated: " + str(self.__att))

    @staticmethod
    def saluto() -> None:
        print("Ciaone")



Test.saluto()
