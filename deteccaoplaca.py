import cv2
import numpy as np
import glob
import math


class NaoOcorre:
    def __init__(self):
        self._00 = 1
        self._01 = 1
        self._10 = 1
        self._11 = 1

class Classe:
    def __init__(self, caractere, n_dim):
        self.caractere = caractere
        self.n_dim = n_dim
        self.n_restricoes = 0
        self.NOC = []
        for i in range(n_dim - 1):
            self.NOC.append(NaoOcorre())
        
class ClassificacaoCaractere:
    def __init__(self,altura,largura,tipo,flag):
        self.altura = altura
        self.largura = largura
        self.tipo = tipo
        self.n_classes = 0
        self.n_dim = altura*largura
        self.classes = []

        if tipo == 1:
            dig_numeros = "0123456789"
            self.n_classes = 10
            for i in range(self.n_classes):
                self.classes.append(Classe(dig_numeros[i],self.n_dim))
            if flag=='S':
                self.monta_arq_aprendizado(dig_numeros,"numeros.png","numeros.txt")
            self.incializa_classificador_2pixels("numeros.txt")
        else:
            dig_letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.n_classes = 26
            for i in range(self.n_classes):
                self.classes.append(Classe(dig_letras[i],self.n_dim))
            if flag=='S':
                self.monta_arq_aprendizado(dig_letras,"letras.png","letras.txt")
            self.incializa_classificador_2pixels("letras.txt")

    def monta_arq_aprendizado(self,digitos,nome_img,nome_arq):
        img = cv2.imread(nome_img)
        arq = open(nome_arq,"w")

        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_cinza, 127, 255, type=cv2.THRESH_BINARY)
        
        contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        lista = []
        for rec in contours:
            x,y,w,h = cv2.boundingRect(rec)
            if h>30 and h<40:
                #cv2.rectangle(img_cinza, (x, y), (x+w-1, y+h-1), (100,100,100), 1)
                lista.append([x, y, x+w-1, y+h-1])
       
        lista.sort()
        for i in range(len(lista)):
            xi = lista[i][0]+1
            yi = lista[i][1]+1
            xf = lista[i][2]-1
            yf = lista[i][3]-1
            img_dig = img_cinza[yi:yf,xi:xf]

            #cv2.imshow('recDig',img_dig)
            #cv2.waitKey(0)
            transicao = self.retornaTransicaoHorizontal(img_dig)
            arq.write(digitos[i]+'|'+transicao+"|\n")
        arq.close()

    def retornaTransicaoHorizontal(self,img):
        dim = (self.largura, self.altura)
        img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        transicao = ""
        #cv2.imshow("img_res",img_res)
     
        flag = True
        for i in range(self.altura):
            if flag:
                for j in range(self.largura):
                    if img_res[i][j] == 0:
                        transicao += '0'
                    else:
                        transicao += '1'
            else:
                for j in range(self.largura-1,-1,-1):
                    if img_res[i][j] == 0:
                        transicao += '0'
                    else:
                        transicao += '1'    
            flag = not flag    
        return transicao


    def incializa_classificador_2pixels(self,nome_arq):
        arq = open( nome_arq,'r')
        todos_dados = arq.readlines()

        pos = 0
        for linha in todos_dados:
            self.classes[pos].n_restricoes = (self.n_dim - 1)*4
            lista = linha.split('|')
            transicao = lista[1]
            for j in range(self.n_dim -1):
                if transicao[j]=='0' and transicao[j+1]=='0' and self.classes[pos].NOC[j]._00 == 1:
                    self.classes[pos].NOC[j]._00 = 0 #0=ocorre, 1=nao ocorre
                    self.classes[pos].n_restricoes-=1;
                if transicao[j]=='0' and transicao[j+1]=='1' and self.classes[pos].NOC[j]._01 == 1:
                    self.classes[pos].NOC[j]._01 = 0 #0=ocorre, 1=nao ocorre
                    self.classes[pos].n_restricoes-=1;
                if transicao[j]=='1' and transicao[j+1]=='0' and self.classes[pos].NOC[j]._10 == 1:
                    self.classes[pos].NOC[j]._10 = 0 #0=ocorre, 1=nao ocorre
                    self.classes[pos].n_restricoes-=1;
                if transicao[j]=='1' and transicao[j+1]=='1' and self.classes[pos].NOC[j]._11 == 1:
                    self.classes[pos].NOC[j]._11 = 0 #0=ocorre, 1=nao ocorre
                    self.classes[pos].n_restricoes-=1;                    
            pos+=1
        arq.close()

    def reconheceCaractereTransicao_2pixels(self, transicao):
        cont_NOC = []
        caractere = ' '

        for i in range(self.n_classes):
            cont_NOC.append(0)
            for j in range(self.n_dim - 1):
                #se tem alguma restricao
                if self.classes[i].NOC[j]._00 == 1 or self.classes[i].NOC[j]._01 == 1 or self.classes[i].NOC[j]._10 == 1 or self.classes[i].NOC[j]._11 == 1:
                    if transicao[j]=='0' and transicao[j+1]=='0' and self.classes[i].NOC[j]._00 == 1:
                        cont_NOC[i]+=1
                    if transicao[j]=='0' and transicao[j+1]=='1' and self.classes[i].NOC[j]._01 == 1:
                        cont_NOC[i]+=1  
                    if transicao[j]=='1' and transicao[j+1]=='0' and self.classes[i].NOC[j]._10 == 1:
                        cont_NOC[i]+=1    
                    if transicao[j]=='1' and transicao[j+1]=='1' and self.classes[i].NOC[j]._11 == 1:
                        cont_NOC[i]+=1

        menor = self.n_dim
        pos = 0
        for i in range(self.n_classes):
            if cont_NOC[i] < menor:
                menor = cont_NOC[i]
                pos = i
        #se empatar, verificar a classe com mais restricoes
        maior = 0
        for i in range(self.n_classes):
            if menor == cont_NOC[i]:
                if self.classes[i].n_restricoes > maior:
                    maior = self.classes[i].n_restricoes
                    pos = i
        if menor < self.n_dim:
            caractere = self.classes[pos].caractere
        return caractere


#imagefiles = glob.glob("Placas de carros/26112002071507.jpg")
imagefiles = glob.glob("Placas de carros/*")
def analisarimagem():
    numPlaca=0
    totalAcertos=0
    for filename in imagefiles:
     
        img = cv2.imread(filename)
        img_copia = img.copy()

        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img_cinza = cv2.GaussianBlur(img_cinza, (3,3), 0.2)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        img_laplacian = cv2.filter2D(img_cinza, -1, kernel) 

        img_gauss = cv2.GaussianBlur(img_laplacian, (3,3), 0)

        ret, thresh = cv2.threshold(img_gauss, 90, 255, type=cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        #obtendo apenas os retangulos de certo tamanho
        lista = []
        for rec in contours:
            x,y,w,h = cv2.boundingRect(rec)
            if h>17 and h<35 and w>3 and w<50:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
                lista.append([x,y,x+w,y+h])

        #eliminando os retangulos sobrepostos
        lista_2 = []
        for i in range(len(lista)):
            flag = 0
            for j in range(len(lista)):
                if i!=j:
                    if lista[i][0]>lista[j][0] and lista[i][0]<lista[j][2] and (lista[i][1]>=lista[j][1] and lista[i][1]<=lista[j][3] or lista[i][3]>=lista[j][1] and lista[i][3]<=lista[j][3]):
                        cv2.rectangle(img, (lista[i][0], lista[i][1]),(lista[i][2], lista[i][3]),(0,255,255),1)                    
                        flag=1
            if flag==0:
                lista_2.append(lista[i])

        #eliminando os retangulos nao alinhados com a maioria
        lista_3 = []
        for i in range(len(lista_2)):
            cont=0
            for j in range(len(lista_2)):
                if i!=j and math.fabs(lista_2[i][1]-lista_2[j][1])<10 and math.fabs(lista_2[i][3]-lista_2[j][3])<10:
                    cont+=1
            if cont>=len(lista_2)*.4:
                lista_3.append(lista_2[i])        


        for i in range(len(lista_3)):
            cv2.rectangle(img, (lista_3[i][0], lista_3[i][1]),(lista_3[i][2], lista_3[i][3]),(0,0,255),3) 

        #obtendo a regiao da placa
        xi=lista_3[0][0]
        yi=lista_3[0][1]
        xf=lista_3[0][2]
        yf=lista_3[0][3]
        for i in range(len(lista_3)):
            if lista_3[i][0]<xi:
                xi=lista_3[i][0]
            if lista_3[i][1]<yi:
                yi=lista_3[i][1]    
            if lista_3[i][2]>xf:
                xf=lista_3[i][2]
            if lista_3[i][3]>yf:
                yf=lista_3[i][3]    
        yi-=7
        yf+=13
        dif=int((330-(xf-xi))/2)
        xi=max(0,xi-dif)
        xf=min(img.shape[1]-1,xf+dif)
        cv2.rectangle(img, (xi,yi),(xf,yf),(0,0,0),3)
    
        #####################################################################################################################
        img_placa=img_copia[yi:yf,xi:xf]
        img_copia = img_placa.copy()
        file_name=filename+"_placa"
        img_placacinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
        
  
    
    #    
        img_gaussian=cv2.GaussianBlur(img_placacinza,(3,3),0)

        kernelx=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewittx=cv2.filter2D(img_gaussian,-1,kernelx)
        prewitty=cv2.filter2D(img_gaussian,-1,kernely)

        img_prewitt=cv2.addWeighted(prewittx,0.5,prewitty,0.5,0)

    
        _,img_thresh = cv2.threshold(img_prewitt, 37, 255, cv2.THRESH_BINARY)
    


    #   ret, thresh = cv2.threshold(img_thresh, 60, 255, type=cv2.THRESH_OTSU)
        #cv2.imshow("img",img_thresh)

    #
        

        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_thresh, contours, -1, (255, 0, 0), 1)
        lista = []
        for rec in contours:
            x,y,w,h = cv2.boundingRect(rec)
            cv2.rectangle(img_placa, (x,y), (x+w, y+h), (255,0,0), 1)
            if h>7 and h<40 and w>7 and w<36:
                cv2.rectangle(img_placa, (x,y), (x+w, y+h), (0,255,0), 2)
                lista.append([x,y,x+w,y+h])
        

        #eliminando os retangulos nao alinhados com a maioria
        lista2 = []
        for i in range(len(lista)):
            cont=0
            for j in range(len(lista)):
                if i!=j and math.fabs(lista[i][1]-lista[j][1])<10 and math.fabs(lista[i][3]-lista[j][3])<12:
                    cont+=1
            if cont>=len(lista)*.4:
                lista2.append(lista[i])        


        for i in range(len(lista2)):
            cv2.rectangle(img_placa, (lista2[i][0], lista2[i][1]),(lista2[i][2], lista2[i][3]),(0,255,0),2) 



        #obtendo a regiao da placa
        xi=lista2[0][0]
        yi=lista2[0][1]
        xf=lista2[0][2]
        yf=lista2[0][3]
        for i in range(len(lista2)):
            if lista2[i][0]<xi:
                xi=lista2[i][0]
            if lista2[i][1]<yi:
                yi=lista2[i][1]    
            if lista2[i][2]>xf:
                xf=lista2[i][2]
            if lista2[i][3]>yf:
                yf=lista2[i][3]    
        if yi-2>=0:
            yi-=2
        yf+=8
        xi-=15
        xf+=12
        cv2.rectangle(img_placa, (xi,yi),(xf,yf),(0,255,0),2)
    #    cv2.imshow(file_name+"t",img_placa)
        ##########
        
        img_placa2=img_copia[yi:yf,xi:xf]
        totalAcertos+=segmentarLetras(img_placa2,filename,xf-xi,yf-yi,numPlaca)
        numPlaca+=1
        #cv2.imshow(file_name,img_placa2)
        
        #cv2.waitKey(0)
    
    print("Total de ACERTOS: "+str(totalAcertos))

def criarListaRetangulos(img_thresh):
    contours, hiearchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lista = []
    for rec in contours:
        x,y,w,h = cv2.boundingRect(rec)
        if h>17 and h<40 and w>5 and w<45:
            lista.append([x, y, x+w-1, y+h-1])
    return lista

def passarLinhaBaixo(lista,img_thresh,col):
    qtde = len(lista)
    img_thresh2=img_thresh.copy()
     #linha branca
    listaY=[]
    for i in range(len(lista)):
        listaY.append(lista[i][3])
    listaY.sort()
   
    media=(int)(len(lista)/2)

    soma=0
    for i in range(media-1,media+2,1):
        soma+=listaY[i]
    soma=(int)(soma/3)
   
    for i in range(col-13):
        img_thresh[soma][i]=255
    lista=criarListaRetangulos(img_thresh)
    if(len(lista)<qtde):
        img_thresh=img_thresh2
    return img_thresh

def segmentarLetras(img,filename,col,lin,numPlaca):
    cv2.imshow("img",img)
    
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("img_cinza",img_cinza)
    img_gaussian=cv2.GaussianBlur(img_cinza,(3,3),0)
  

    _,img_thresh = cv2.threshold(img_gaussian, 50, 255, cv2.THRESH_OTSU)
    
    img_thresh2=img_thresh.copy()
    lista=criarListaRetangulos(img_thresh)
   
    img_thresh=passarLinhaBaixo(lista,img_thresh,col)
    
   

    
    lista=criarListaRetangulos(img_thresh)
    gamma=0.4
    if len(lista)<7 :
       
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
       
        img_gaussian2=cv2.LUT(img_gaussian, table)
        #cv2.imshow("img_gamma",img_gaussian)
        _,img_thresh = cv2.threshold(img_gaussian2, 10, 255, cv2.THRESH_OTSU)

        img_thresh=passarLinhaBaixo(lista,img_thresh,col)
       

    cv2.imshow("img_thresh",img_thresh)
    
    
    
    contours, hiearchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lista = []
    for rec in contours:
        x,y,w,h = cv2.boundingRect(rec)
        if h>15 and h<40 and w>5 and w<45:
            lista.append([x, y, x+w-1, y+h-1])

    #eliminando os retangulos sobrepostos
    if(len(lista)>7):
        lista_2 = []
        for i in range(len(lista)):
            flag = 0
            for j in range(len(lista)):
                if i!=j:
                    if lista[i][0]>lista[j][0] and lista[i][0]<lista[j][2] and (lista[i][1]>=lista[j][1] and lista[i][1]<=lista[j][3] or lista[i][3]>=lista[j][1] and lista[i][3]<=lista[j][3]):                   
                        flag=1
            if flag==0:
                cv2.rectangle(img, (lista[i][0],lista[i][1]), (lista[i][2], lista[i][3]), (0,255,0), 2)
                lista_2.append(lista[i])
        

        lista=lista_2
    else:
        lista = []
        for rec in contours:
            x,y,w,h = cv2.boundingRect(rec)
            if h>15 and h<40 and w>5 and w<45:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                lista.append([x, y, x+w-1, y+h-1])
    cv2.imshow("img",img)


    #Colocando todas as placas
    listaPlacas=[]
    listaPlacas.append(['L','O','H',8,5,1,6])
    listaPlacas.append(['L','I','D',5,8,1,6])
    listaPlacas.append(['L','N','Z',7,6,7,5])
    listaPlacas.append(['L','O','F',2,0,2,5])
    listaPlacas.append(['L','A','Y',2,5,9,6])
    listaPlacas.append(['L','O','A',2,5,6,3])
    listaPlacas.append(['G','O','L',0,1,9,7])
    listaPlacas.append(['L','N','U',2,8,8,5])
    listaPlacas.append(['L','K','I',6,5,4,9])
    listaPlacas.append(['K','R','D',7,3,6,6])
    listaPlacas.append(['L','N','J',8,7,5,3])
    listaPlacas.append(['K','U','F',2,3,7,6])
    listaPlacas.append(['L','N','P',1,4,1,7])
    listaPlacas.append(['D','U','R',7,7,7,8])
    listaPlacas.append(['L','N','F',5,8,7,3])
    listaPlacas.append(['L','N','R',3,8,3,9])
    listaPlacas.append(['L','I','N',2,1,4,8])
    listaPlacas.append(['K','M','V',8,0,6,8])
    listaPlacas.append(['L','N','O',3,9,6,8])
    listaPlacas.append(['L','N','P',7,2,2,1])
    listaPlacas.append(['K','M','N',0,8,5,7])
    listaPlacas.append(['L','N','D',3,4,1,6])
    listaPlacas.append(['L','N','Z',3,7,7,7])
    listaPlacas.append(['L','N','G',6,6,3,1])
    listaPlacas.append(['K','N','I',0,5,0,8])
    listaPlacas.append(['K','M','U',0,0,8,7])
    listaPlacas.append(['L','O','E',2,0,2,2])
    listaPlacas.append(['L','C','S',8,2,1,7])
    listaPlacas.append(['L','B','D',2,3,1,9])
    listaPlacas.append(['C','S','E',9,7,8,0])
    listaPlacas.append(['L','A','G',5,8,8,7])
    listaPlacas.append(['L','B','U',5,6,9,9])
    listaPlacas.append(['L','B','L',9,6,0,6])
    listaPlacas.append(['L','A','K',8,6,1,4])
    listaPlacas.append(['L','N','I',2,9,9,3])
    listaPlacas.append(['L','O','E',0,8,3,6])
    listaPlacas.append(['K','Q','H',7,7,3,1])
    listaPlacas.append(['A','A','Y',5,1,2,7])
    listaPlacas.append(['L','O','A',7,9,9,3])
    listaPlacas.append(['K','N','D',8,8,6,5])
    listaPlacas.append(['L','N','R',0,4,9,1])
    listaPlacas.append(['K','N','P',9,8,3,7])
    listaPlacas.append(['L','A','V',5,2,3,3])
    listaPlacas.append(['L','O','A',1,7,8,4])
    listaPlacas.append(['K','M','G',6,9,2,1])
    listaPlacas.append(['L','B','O',1,2,3,0])
    listaPlacas.append(['K','O','D',7,3,4,7])
    listaPlacas.append(['L','C','N',7,7,2,4])
    listaPlacas.append(['A','B','Y',9,1,9,8])
    listaPlacas.append(['G','Z','A',2,4,7,7])
    listaPlacas.append(['L','N','Z',1,7,2,4])
    listaPlacas.append(['L','A','R',7,2,0,7])
    listaPlacas.append(['K','M','O',7,2,6,5])
    listaPlacas.append(['C','T','B',1,5,6,9])
    listaPlacas.append(['L','O','F',3,4,0,5])
    listaPlacas.append(['L','I','K',3,9,8,8])
    listaPlacas.append(['L','C','D',7,6,4,5])
    listaPlacas.append(['L','B','L',9,7,0,9])
    listaPlacas.append(['L','N','Z',4,2,9,2])
    listaPlacas.append(['J','L','W',9,4,6,4])
    
    acerto=0

    print(filename)
    lista.sort()
    for i in range(len(lista)):
        xi = lista[i][0]+1
        yi = lista[i][1]+1
        xf = lista[i][2]-1
        yf = lista[i][3]
        img_dig = img_thresh[yi:yf,xi:xf]
        
        cv2.imshow("dig",img_dig)
        
        if i<=2:
            transicao = cl_letra.retornaTransicaoHorizontal(img_dig)
            digito=cl_letra.reconheceCaractereTransicao_2pixels(transicao)
            print(digito)
        else:
            transicao = cl_numero.retornaTransicaoHorizontal(img_dig)
            digito=int(cl_numero.reconheceCaractereTransicao_2pixels(transicao))
            print(digito)

        if digito==listaPlacas[numPlaca][i]:
            acerto+=1
    
    print("Acerto na placa: "+filename+" = "+str(acerto))
    #cv2.waitKey(0)
    return acerto

   
    
    

if __name__ == "__main__":
    cl_letra = ClassificacaoCaractere(45,55,2,'S')
    cl_numero = ClassificacaoCaractere(45,55,1,'S')
    analisarimagem()
 
