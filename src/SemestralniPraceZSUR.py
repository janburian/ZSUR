# Importy modulů
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math as math

# POMOCNE METODY (Eukleid, načítání soboru, matice vzdálenosti)
# Metoda nacita soubor s daty
def load(infile):
    fin = open(infile, "rt")
    poleVektoru = []
    for line in fin:
        souradniceX, souradniceY = line.split()
        v = tuple([float(souradniceX), float(souradniceY)])
        poleVektoru.append(v)
    fin.close()
    print("Data v pořádku načtena.")
    return poleVektoru

# Metoda, ktera pocita a vraci vzdalenost mezi jednotlivymi prvky
def vypoctiEukleidVzdalenost(vektor_X, vektor_Y):
    rozdilX = vektor_X[0] - vektor_Y[0]
    rozdilY = vektor_X[1] - vektor_Y[1]
    Eukleid_X = rozdilX ** 2
    Eukleid_Y = rozdilY ** 2
    dist = Eukleid_X + Eukleid_Y
    return dist

# Metoda pocita a vraci matici vzdalenosti
def vypoctiMaticiVzdalenosti(data):
    print("Počítám matici vzdálenosti.")
    dist = 0
    n = len(data)
    matrix = np.zeros((n,n))

    for i in range(len(matrix)):  # vypocet matice vzdalenosti (horni trojuhelnik)
        for j in range(i + 1, len(matrix)):
            matrix[i, j] = vypoctiEukleidVzdalenost(data[i], data[j])
    matrix = np.triu(matrix) + np.tril(matrix.T, 1)  # spojeni horni a dolni trojuhelnikove matice
    print("Matice vzdálenosti vypočtena.")
    return matrix
     

# ZJIŠTĚNÍ POČTU TŘÍD
# Aglomerativni metoda
def shlukovaHladina(maticeVzdalenosti):
    hladiny = []
    pocetTrid = 1

    while(len(maticeVzdalenosti) != 1):
        min_vzdalenost = (np.amin(maticeVzdalenosti[np.nonzero(maticeVzdalenosti)]))
        hladiny.append(min_vzdalenost)

        poleIndexu_min = np.where(maticeVzdalenosti == min_vzdalenost)
        
        pole1 = poleIndexu_min[0]
        pole2 = poleIndexu_min[1]

        minIndex1 = [pole1[0], pole2[0]]
        minIndex2 = [pole2[0], pole1[0]]

        line1 = maticeVzdalenosti[minIndex1[0], :]    
        line2 = maticeVzdalenosti[minIndex2[0], :]

        column1 = maticeVzdalenosti[:, minIndex1[1]]
        column2 = maticeVzdalenosti[:, minIndex2[1]]

        lineFinal = np.minimum(line1, line2)
        columnFinal = np.minimum(column1, column2)

        maticeVzdalenosti[minIndex1[0], :] = lineFinal
        maticeVzdalenosti[:, minIndex1[0]] = columnFinal
  
        maticeVzdalenosti = np.delete(maticeVzdalenosti, minIndex1[1], 0) 
        maticeVzdalenosti = np.delete(maticeVzdalenosti, minIndex1[1], 1)

        print()
        print(maticeVzdalenosti)

    for i in range(len(hladiny)):
        if(hladiny[i] > (0.2 * max(hladiny))): # dostatecny rozdil 
            pocetTrid += 1

    print("Shluková hladina - počet tříd je: " + str(pocetTrid))
    return pocetTrid

# Metoda retezove mapy
def retezovaMapa(maticeVzdalenosti, index_pocatek):
    #bod_pocatek = data[index_pocatek]
    matice = maticeVzdalenosti
    index = index_pocatek
    pocetTrid = 1
    listMinVzdalenosti = []
    konstanta = 70 #pokud vzdalenost vetsi nez tato konstanta -> dalsi trida
    retezova_mapa = {index+1: 0}
    vektory_postupne = []
    #vektory_postupne.append(bod_pocatek)

    for i in range(len(matice)-1):
        line = matice[index, :] 
        min_vzdalenost = (np.amin(line[np.nonzero(line)]))
        listMinVzdalenosti.append(min_vzdalenost)
        indexRadku_min = np.where(line == min_vzdalenost)
        test = indexRadku_min[0][0]
        next = int(indexRadku_min[0][0]) + 1
        vektory_postupne.append(data[next-1])
        matice[index, :] = 0
        matice[:, index] = 0
        retezova_mapa.update({next: min_vzdalenost})
        index = int(indexRadku_min[0][0])

    listMinVzdalenosti.sort(reverse = True) # serazeni od nejvetsi vzdalenosti po nejmensi
    maxVzdal = listMinVzdalenosti[0]
    prumer_vzdalenost = np.mean(listMinVzdalenosti)

    dostatecnaVzdalenost = 47
    for i in range(len(listMinVzdalenosti)-1):
        if((listMinVzdalenosti[i] - listMinVzdalenosti[i+1]) > dostatecnaVzdalenost): # dostatecny rozdil mezi vzdalenostmi
            pocetTrid += 1

    return pocetTrid, vektory_postupne

# Metoda MAXIMIN
def MAXIMIN(maticeVzdalenosti, index_pocatek, q):
    A = maticeVzdalenosti
    stredy = []
    listVzdalenostiPrvkuKeStredum = []
    slovnik = {}
    
    listNejkratsiVzdalenosti = []
    line = A[index_pocatek, :] # startovni radek a pote radek, s nejvetsi vzdalenosti
    max_vzdalenost = (np.amax(line[np.nonzero(line)]))
    indexRadku_max = np.where(line == max_vzdalenost)
    index_data = indexRadku_max[0][0]

    stredy.append(data[index_pocatek])
    stredy.append(data[index_data])

    switch = 1 # promenna slouzici k zastaveni cyklu while
    while switch == 1: 
        listNejkratsiVzdalenosti.clear()
        for i in range(len(A)): # Vzdalenosti ke stredum 
            bod = data[i]
            for j in range(len(stredy)):
                stred = stredy[j]
                dist = vypoctiEukleidVzdalenost(bod, stred)
                listVzdalenostiPrvkuKeStredum.append(dist) # pomocny seznam
            listNejkratsiVzdalenosti.append(min(listVzdalenostiPrvkuKeStredum))
            slovnik.update({i: min(listVzdalenostiPrvkuKeStredum)})
            listVzdalenostiPrvkuKeStredum.clear()

        for i in range(len(stredy)):
            listNejkratsiVzdalenosti.remove(0)

        dMax = max(listNejkratsiVzdalenosti)
        listVzdalenostiStredu = []
        for u in range(len(stredy)-1):
            stred1 = stredy[u] 
            for e in range(u+1, len(stredy)):
                stred2 = stredy[e]
                dist = vypoctiEukleidVzdalenost(stred1, stred2)
                listVzdalenostiStredu.append(dist)

        vypocet = 0
        soucetVzdalenosti = 0
        for i in range(len(listVzdalenostiStredu)):
            soucetVzdalenosti += listVzdalenostiStredu[i]
        vypocet = q * (1/(len(stredy))) * soucetVzdalenosti 

        if (dMax > vypocet): 
            values = slovnik.values()
            maximum = max(values) # maximum z minimalnich vzdalenosti
            index_novy_stred = list(slovnik.keys())[list(slovnik.values()).index(maximum)]
            stredy.append(data[index_novy_stred])
            switch = 1
        else:
            switch = 0

    pocetTrid = len(stredy)
    print("MAXIMIN - počet tříd je: " + str(pocetTrid))
    return pocetTrid


# ROZDĚLENÍ DAT DO JEDNOTLIVÝCH TŘÍD
# Pomocna metoda k metode k-means
def vypocitejMi(seznamIndexu, data):
    miX = 0
    miY = 0
    for i in range(len(seznamIndexu)):
        vektor = data[seznamIndexu[i]]
        miX += vektor[0]
        miY += vektor[1]
    miX = miX / len(seznamIndexu) 
    miY = miY / len(seznamIndexu)
    mi = (miX, miY)
    return mi

# Metoda K-means 
def k_means(R, data):
    stredy_zacatek = [] # pocet stredu je zavisly na poctu trid neboli R = pocet stredu
    listVzdalenostiBoduKeStredum = []
    finalListMinVzdalenosti = []
    stredy = []
    listTrid = []

    for i in range(R):
        #randomIndex = rand.randint(0, len(data)) # nahodne stredy obcas zlobi, vygenerujou se spatne stredy
        #stredy_zacatek.append(data[randomIndex])
        stredy_zacatek.append(data[i])

    stredy = stredy_zacatek 

    switch = 1
    while switch == 1:
        finalListMinVzdalenosti.clear()
        listMi = []
        J = 0
        J_previous = J
        

        for i in range(R):
            listTrid.append([]) # list listu na kazdem indexu je pridan dalsi list len(listTrid) = pocet trid

        for i in range(len(data)):
                bod = data[i] 
                for j in range(len(stredy)):
                    stred = stredy[j]
                    dist = vypoctiEukleidVzdalenost(bod, stred)
                    listVzdalenostiBoduKeStredum.append((dist, j)) # tuple -  vzdalenost a index sloupce
                minDist = min(listVzdalenostiBoduKeStredum)
                finalListMinVzdalenosti.append(minDist)
                listTrid[minDist[1]].append(data[i])
                listVzdalenostiBoduKeStredum.clear()
    
        pomocne_NoveStredy = []
        for i in range(0, R):
            pomocne_NoveStredy.append([])

        for i in range(len(finalListMinVzdalenosti)):
            tuple_minVzdalenost_sloupec = finalListMinVzdalenosti[i] 
            index = tuple_minVzdalenost_sloupec[1]
            J += tuple_minVzdalenost_sloupec[0]
            pomocne_NoveStredy[index].append(i)

        print("Kriterialni hodnota - k-means: J = {}".format(J))

        listMi = [] # nove stredy
        for i in range(len(pomocne_NoveStredy)):
            mi = vypocitejMi(pomocne_NoveStredy[i], data)
            listMi.append(mi)

        if (stredy != listMi):
            if(J_previous == J):
                stredy == listMi
            if(J_previous * 0.95 < J):
                switch = 1
                listTrid.clear()
            stredy = listMi
        else: 
            switch = 0

    print("HOTOVO - Data rozdělena pomocí k-means.")
    return [listTrid, listMi]

# Pomocna metoda K-means pro binarni deleni 
def k_means2(R, data, stredy):
    listVzdalenostiBoduKeStredum = []
    finalListMinVzdalenosti = []
    listTrid = []
    list_J = []
    counter = 0

    switch = 1
    while switch == 1:
        list_J.clear()
        listTrid.clear()
        finalListMinVzdalenosti.clear()
        listMi = []
        J = 0
        J_previous = J

        for i in range(R):
            list_J.append([])

        for i in range(R):
            listTrid.append([]) # list listu na kazdem indexu je pridan dalsi list len(listTrid) = pocet trid

        for i in range(len(data)):
                bod = data[i] 
                for j in range(len(stredy)):
                    stred = stredy[j]
                    dist = vypoctiEukleidVzdalenost(bod, stred)
                    listVzdalenostiBoduKeStredum.append((dist, j)) # tuple -  vzdalenost a index sloupce
                minDist = min(listVzdalenostiBoduKeStredum)
                finalListMinVzdalenosti.append(minDist)
                listTrid[minDist[1]].append(data[i])
                listVzdalenostiBoduKeStredum.clear()
    
        pomocne_NoveStredy = []
        for i in range(0, R):
            pomocne_NoveStredy.append([])

        for i in range(len(finalListMinVzdalenosti)):
            tuple_minVzdalenost_sloupec = finalListMinVzdalenosti[i] 
            index = tuple_minVzdalenost_sloupec[1]
            J = tuple_minVzdalenost_sloupec[0]
            list_J[index].append(J)
            pomocne_NoveStredy[index].append(i)
            J = 0

        listMi = [] # nove stredy
        for i in range(len(pomocne_NoveStredy)):
            mi = vypocitejMi(pomocne_NoveStredy[i], data)
            listMi.append(mi)

        #finalListJ = []
        #for i in range(len(list_J)):
            #J_pomocne = list_J[i]
            #for j in range(len(J_pomocne)):
                #J += J_pomocne[j]
         
        counter += 1

        if (stredy != listMi):
            stredy = listMi
            #if(J_previous == J and counter > 0):
                #stredy == listMi
            #if(J_previous * 0.95 < J and counter > 0):
                #switch = 0
                #listTrid.clear()
            #stredy = listMi
        else: 
            switch = 0
            return (listTrid, list_J, stredy)

# Metoda - binarni deleni - nerovnomerne
def binarniDeleni_nerovnomerne(maticeVzdalenosti, R, data):
    stredy = []
    finalRozdeleniTrid = []
    max_vzdalenost1 = (np.amax(maticeVzdalenosti[np.nonzero(maticeVzdalenosti)]))
    poleIndexu_max = np.where(maticeVzdalenosti == max_vzdalenost1)
    stredy.append(data[poleIndexu_max[0][0]])
    stredy.append(data[poleIndexu_max[1][0]])

    poleIndexu_max = []
    tridy = []
    counter = 0
    J_list = []
    J_list1_final = []
    J_final = [] # na jednotlivych indexech 0, 1, ...n je J1, J2, ...Jn
    trida_deleni = data
    stredy_z_k_means = []

    if(R == 2): # pokud jen 2 tridy
        hodnoty = k_means2(2, data, stredy)
        tridy = hodnoty[0]
        finalRozdeleniTrid = tridy
        print("HOTOVO - Data rozdělena pomocí nerovnoměrného binárního dělení.")
        return finalRozdeleniTrid

    while (len(tridy) <= R):
        hodnoty = []
        listVzdalenosti = []
        J_list1_final.clear()
        
        hodnoty = k_means2(2, trida_deleni, stredy) # rozdeleni, J_list a vysledne stredy z k-means
        stredy_z_k_means.append(hodnoty[2])
        
        if(counter > 0):
            J_final[indexHorsiJ] = 0

        if(counter == 0): 
            tridy = hodnoty[0]
        else:
            tridy[indexHorsiJ] = 0
            tridy.append(hodnoty[0][0])
            tridy.append(hodnoty[0][1])

        pocetTrid = 0
        for i in range(len(tridy)):
            if(tridy[i] == 0):
                continue
            pocetTrid += 1

        if(pocetTrid == R):
            for i in range(len(tridy)):
                if (tridy[i] == 0):
                    continue
                finalRozdeleniTrid.append(tridy[i])
            print("HOTOVO - Data rozdělena pomocí nerovnoměrného binárního dělení.")
            return finalRozdeleniTrid, stredy_z_k_means
            
        J_list = hodnoty[1]
    
        for i in range(len(J_list)):
            J_final.append(sum(J_list[i]))

        horsiJ = max(J_final)
        indexHorsiJ = J_final.index(horsiJ)

        trida_deleni = tridy[indexHorsiJ]
        tridy[indexHorsiJ] = 0

        # Nový výpočet vzdáleností
        for i in range(len(trida_deleni)):
            bod1 = trida_deleni[i]
            for j in range(len(trida_deleni)):
                bod2 = trida_deleni[j]
                vzdalenost = vypoctiEukleidVzdalenost(bod1, bod2)
                listVzdalenosti.append(vzdalenost)

        max_vzdalenost = max(listVzdalenosti)
        index = np.where(maticeVzdalenosti == max_vzdalenost)
        
        newStredy = []

        newStredy.append(data[index[0][0]])
        newStredy.append(data[index[1][0]])
        stredy = newStredy

        counter += 1
    return tridy

def vypoctiKriterialniFunkci(trida, stred):
    J = 0
    for vektor in trida:
        dist = vypoctiEukleidVzdalenost(vektor, stred)
        J += dist
    return J

# Metoda - iterativni optimalizace
def iterativniOptimalizace(listTrid, stredy, data ):
    for i in range(len(data)):
        listA = []
        newStredy = []
        A = 0
        vzdalenost = 0
        pomocna = 0
        minimum = 0
        pozicePrvku = 0

        for k in range(len(listTrid)):
            if(data[i] in listTrid[k]):
                pozicePrvku = k 

        if((data[i] in listTrid[pozicePrvku]) and (len(listTrid[pozicePrvku]) == 1)):
            continue

        for j in range(len(stredy)):
            trida = listTrid[j]
            J_k = vypoctiKriterialniFunkci(trida, stredy[j])
            print("J({}) = {}".format(j+1,J_k))
            vzdalenost = vypoctiEukleidVzdalenost(data[i], stredy[j])
            if data[i] in trida:
                A = (len(listTrid[pozicePrvku])/((len(listTrid[pozicePrvku])) - 1)) * vzdalenost # výpočet vzdálenosti, pokud prvek je v dané třídě
                listA.append(A) 
                pomocna = A
            else:
                A = (len(trida)/((len(trida)) + 1)) * vzdalenost # výpočet vzdálenosti, pokud prvek není v dané třídě
                listA.append(A)
        print("Iterativní optimalizace...")
           
        listFinalA = listA.copy()   
        minimum = min(listA)
        listA.remove(pomocna)
        indexMinima = listFinalA.index(minimum)
        indexPomocna = listFinalA.index(pomocna)
   
        if(pomocna != minimum):
            if(len(listTrid[indexPomocna]) > 1):
                listTrid[indexPomocna].remove(data[i])
                listTrid[indexMinima].append(data[i])

            for u in range(len(stredy)): # výpočet nových středů
                vektory = listTrid[u]
                X = 0
                Y = 0
                for y in range(len(vektory)): 
                    vektor = vektory[y]
                    X += vektor[0]
                    Y += vektor[1]
                X /= len(vektory)
                Y /= len(vektory)
                novyStred = (X, Y)
                newStredy.append(novyStred)
            stredy = newStredy
    print("Iterativní optimalizace v pořádku proběhla.")
    return listTrid


# KLASIFIKÁTORY
# Bayesovský klasifikátor
def vynasobVektory(vektor1, vektor2):
    a11 = vektor1[0] * vektor2[0]
    a12 = vektor1[0] * vektor2[1]
    a21 = vektor1[1] * vektor2[0]
    a22 = vektor1[1] * vektor2[1]

    return np.array([[a11, a12], [a21, a22]])

def vypoctiRozdilVektoru(vektor, stred):
    rozdil_X = vektor[0] - stred[0]
    rozdil_Y = vektor[1] - stred[1]
    return (rozdil_X, rozdil_Y)

def vypoctiKovariancniMatice(rozdeleni_k_means, stredy_k_means):
    sum = 0 
    listKovariancnichMatic = []
    index = 0
    for trida in rozdeleni_k_means:
        stred = stredy_k_means[index]
        for vektor in trida:
            vektor = vypoctiRozdilVektoru(vektor, stred)
            soucin = vynasobVektory(vektor, vektor)
            sum += soucin
        index += 1

        listKovariancnichMatic.append((sum)/len(trida))
        sum = 0
    return listKovariancnichMatic

def vypoctiApriorniPravdepododnostiTrid(rozdeleni_k_means, data):
    list_apr_ppst_tridy = []
    pocetDat = len(data)
    for trida in rozdeleni_k_means:
        delkaTridy = len(trida)
        list_apr_ppst_tridy.append(delkaTridy / pocetDat)

    return list_apr_ppst_tridy

def vynasobVektorTInvKovMaticiVektor(rozdil_vektor_stred, inv_kov_matice):
    skalar = 0
    #rozdil_vektor_stred = np.array([-1,0])
    #inv_kov_matice = np.array([[2,1], [3,4]])
    transpVektor = np.transpose(rozdil_vektor_stred)
    soucinTranspVektorKovMatice = np.dot(transpVektor, inv_kov_matice)
    skalar = np.dot(soucinTranspVektorKovMatice, rozdil_vektor_stred)

    return skalar

def rozhodniOTride(list_apr_ppst_tridy, list_Pi_omega):
    listSoucinuPpsti = []
    for aprPpst, Pi_omega in zip(list_apr_ppst_tridy, list_Pi_omega):
        listSoucinuPpsti.append(aprPpst * Pi_omega)

    maximum = max(listSoucinuPpsti)
    indexMaxima = listSoucinuPpsti.index(maximum)

    return indexMaxima

def odhadniStred(trida):
    X = 0
    Y = 0
    for vektor in trida:
        X += vektor[0]
        Y += vektor[1]
    X /= len(trida)
    Y /= len(trida)
    stred = (X, Y)
    return stred

def vzorecNormalniRozdeleni_Bayes(dimenze, kovariancniMatice, stred, x):
    det_kov_matice = np.linalg.det(kovariancniMatice) 
    vzorec_cast1 = 1/(math.sqrt(((2 * math.pi) ** dimenze) * det_kov_matice))
    
    rozdil_vektor_stred = np.array(vypoctiRozdilVektoru(x, stred))
    inv_kov_matice = np.linalg.inv(kovariancniMatice)
    soucin = vynasobVektorTInvKovMaticiVektor(rozdil_vektor_stred, inv_kov_matice)
    
    vzorec_cast2 = math.exp(-0.5 * soucin) 
    Pi_omega = (vzorec_cast1 * vzorec_cast2)
    return Pi_omega

def Bayesovsky_klasifikator(stredy, rozdeleni_k_means, list_apr_ppst_tridy, list_kovariancniMatice, data, x):
    dimenze = 2 
    points = []
    index = 0

    list_Pi_omega = []
    vzorec_final = 0

    for i in range(len(rozdeleni_k_means)):
        Pi_omega = vzorecNormalniRozdeleni_Bayes(dimenze, list_kovariancniMatice[i], stredy[i], x)
        list_Pi_omega.append(Pi_omega)

    trida = rozhodniOTride(list_apr_ppst_tridy, list_Pi_omega)
    return trida


# Klasifikátor - nejbližší soused
def knn(rozdeleni_z_kmeans, vektor_klasifikace, K):
    pocetTrid = len(rozdeleni_z_kmeans)
    listVzdalenosti = []
    indexyNejblizsichSousedu = []
    listMinDist = []

    for i in range(pocetTrid):
        listVzdalenosti.append([])

    for i in range(pocetTrid):
        trida = rozdeleni_z_kmeans[i]
        for j in range(len(trida)):
            vektor = trida[j]
            distance = vypoctiEukleidVzdalenost(vektor_klasifikace, vektor)
            listVzdalenosti[i].append(distance)
        minDist = min(listVzdalenosti[i]) # nejmensi vzdalenost v danem konkretnim shluku
        listMinDist.append(minDist)

    if(K == 1):
        finalMinDist = min(listMinDist)
        indexTridyMin = listMinDist.index(finalMinDist) # zjistim, tridu, ve ktere je nejblizsi bod, cili kam budu klasifikovat bod (reseni pro 1 nejblizsiho souseda)
        return indexTridyMin

    else:
        listVzdalenostiFinal = []

        for i in range(len(listVzdalenosti)):
            listVzdalenosti[i].sort()

        pomocnyListSousede = []
        for i in range(len(listVzdalenosti)):
            pomocnyListSousede.append([])

        for i in range(len(listVzdalenosti)):
            shluk = listVzdalenosti[i]
            for j in range(K):
                dist = shluk[j]
                pomocnyListSousede[i].append(dist)

        finalListVzdalenosti = []
        for i in range(len(pomocnyListSousede)):
            vzdalenosti = pomocnyListSousede[i]
            soucetVzdal = sum(vzdalenosti)
            finalListVzdalenosti.append(soucetVzdal)

        minDist = min(finalListVzdalenosti)
        indexTridy = finalListVzdalenosti.index(minDist) # Index tridy, do ktere dany vektor patri
        return indexTridy

# Vektorová kvantizace pro nerovnoměrné dělení 
def vektorovaKvantizace_nerov(vektor_vekt_kvantizace, stredy_vekt_kvantizace):
    # Pro nerovnomerne deleni 
    for i in range(len(stredy_vekt_kvantizace)):
        stredy = stredy_vekt_kvantizace[i]
        for j in range(len(stredy)):
            dist = math.sqrt(vypoctiEukleidVzdalenost(vektor_vekt_kvantizace, stredy[j]))
            listVzdalenosti.append((dist, counterStredu))
            counterStredu += 1
        minDist = min(listVzdalenosti)
        listMinVzdalenosti.append(minDist)
        listVzdalenosti.clear()
    
    finalMin = min(listMinVzdalenosti)
    indexTridy = finalMin[1]

# Vektorová kvantizace pro rovnoměrné dělení 
def vektorovaKvantizace_rov(vektor_vekt_kvantizace, stredy_vekt_kvantizace, startIndex = 0):
    listVzdalenosti = []
    listMinVzdalenosti = []
    counter = 0
    
    stredy = stredy_vekt_kvantizace[startIndex]
    # Pro rovnomerne deleni
    for i in range(len(stredy)):
        dist = vypoctiEukleidVzdalenost(vektor_vekt_kvantizace, stredy[i])
        listVzdalenosti.append((dist, i))
    minDist = min(listVzdalenosti)
    indexTridy = minDist[1]

    nextIndex = 2 * startIndex + indexTridy + 1
    if (nextIndex >= len(stredy_vekt_kvantizace)):
        return stredy_vekt_kvantizace[indexTridy]

    vektorovaKvantizace_rov(vektor_vekt_kvantizace, stredy_vekt_kvantizace, nextIndex)

    print()

# Vektorová kvantizace K-means
def vektorovaKvantizace_Kmeans(vektor, stredyZKmeans, rozdeleniKmeans):
    listSmallestDist = [] 
    for i in range(len(stredyZKmeans)):
        dist = vypoctiEukleidVzdalenost(vektor, stredyZKmeans[i])
        listSmallestDist.append(dist)

    minDist = min(listSmallestDist)
    indexTridy = listSmallestDist.index(minDist)
    return indexTridy
    print()

# Rosenblattův algoritmus
def dosazeniDoQ(q, vektor):
    g = 0
    k = q[0]
    kx = q[1] * vektor[0]
    ky = q[2] * vektor[1]
    return k + kx + ky
    
def sgn(vysledekDosazeni):
    if (vysledekDosazeni < 0):
        return -1
    elif(vysledekDosazeni == 0):
        return 0
    else:
        return 1

def kontrolaOK(vektory, omegy, q):
    result = []
    for i in range(len(vektory)):
        vysledekDosazeni = dosazeniDoQ(q, vektory[i])
        signum = sgn(vysledekDosazeni)
        result.append(signum * omegy[i] == 1)
    return result

def jsouVsechnyOk(okList):
    for i in range(len(okList)):
        if (okList[i] != True):
            return False
    return True

def vypoctiSoucinQ_vektorRozsireny(q, obrazovyVektor_rozsireny, omega):
    mezivypocet1 = q[0] * obrazovyVektor_rozsireny[0]
    mezivypocet2 = q[1] * obrazovyVektor_rozsireny[1]
    mezivypocet3 = q[2] * obrazovyVektor_rozsireny[2]

    return ((mezivypocet1 + mezivypocet2 + mezivypocet3) * omega)

def vypoctiNoveQ(q, obrazovyVektor_rozsireny, omega, c_k):
    mezivypocet1 = (q[0] + (omega * c_k * obrazovyVektor_rozsireny[0])) 
    mezivypocet2 = (q[1] + (omega * c_k * obrazovyVektor_rozsireny[1])) 
    mezivypocet3 = (q[2] + (omega * c_k * obrazovyVektor_rozsireny[2])) 

    return (mezivypocet1, mezivypocet2, mezivypocet3) 

def spocitejY(q_final, X):
    Y = (X * (-q_final[1]) - q_final[0])/q_final[2]
    return Y

def spocitejUsecku(q_final):
    Y0 = spocitejY(q_final, -1000)
    Y1 = spocitejY(q_final, 1000)

    return ([-1000, 1000], [Y0, Y1])

def Rosenblattuv_algoritmus(rozdeleni_Kmeans, q):
    listOmega = [1, -1]
    okList = []

    pomocnyListOmega = []
    listVektoru = []
    for i in range(len(rozdeleni_Kmeans)):
        trida = rozdeleni_Kmeans[i]
        for vektor in trida:
            listVektoru.append(vektor)
            pomocnyListOmega.append(listOmega[i])

    return Rosenblatt_pomocna(listVektoru, pomocnyListOmega, q)

def Rosenblatt_pomocna(listVektoru, pomocnyListOmega, q):
    for idx_Ros in range(500000): # zamezeni nekonecneho cyklu
        kontrola = kontrolaOK(listVektoru, pomocnyListOmega, q)
        if (jsouVsechnyOk(kontrola)):
            print("Počet iterací - Rosenblattův algoritmus: {}".format(idx_Ros + 1))
            return q

        mamNoveQ = False
        for i in range(len(listVektoru)):
            vektor = listVektoru[i]
            obrazovyVektor_rozsireny = [1]
            for elem in vektor:
                obrazovyVektor_rozsireny.append(elem) 

            obrazovyVektor_rozsireny = tuple(obrazovyVektor_rozsireny)

            soucin_Q_vektorRozsireny = vypoctiSoucinQ_vektorRozsireny(q, obrazovyVektor_rozsireny, pomocnyListOmega[i])
            if(soucin_Q_vektorRozsireny < 0):
                q = vypoctiNoveQ(q, obrazovyVektor_rozsireny, pomocnyListOmega[i], 1)
                mamNoveQ = True

        if (not mamNoveQ):
            print("Počet iterací - Rosenblattův algoritmus: {}".format(idx_Ros + 1))
            return q

    print("Počet iterací - Rosenblattův algoritmus: {}".format(idx_Ros + 1))
    return q

def SpojTridy(trida1, trida2):
    result = []
    for x in trida1:
        result.append(x)
    for x in trida2:
        result.append(x)
    return result

def ZjistiDeliciPrimkuRosenblatt(trida1, trida2, q0):
    data_Rosenblatt = []

    data_Rosenblatt.append(trida1)
    data_Rosenblatt.append(trida2)
    
    q_final = Rosenblattuv_algoritmus(data_Rosenblatt, q0)
    body_Usecka = spocitejUsecku(q_final)
    return body_Usecka, q_final

def zjistiOmega(q, bod):
    X_bod = bod[0]
    Y_bod = bod[1]

    yProQ = spocitejY(q, X_bod)
    if(Y_bod > yProQ):
        return 1
    return -1

def zjistiSegmentProBod(bod, listPrimekQ):
    segment = []
    for primkaQ in listPrimekQ:
        omega = zjistiOmega(primkaQ, bod)
        segment.append(omega)
    return segment

def jsouStejneSegmenty(segmentA, segmentB):
    for i in range(len(segmentA)):
        if(segmentA[i] != segmentB[i]):
            return False
    return True

def patriBodDoTridy(segmentBod, segmentyTrid):
    for segmentTridy in segmentyTrid:
        if(segmentBod == segmentTridy):
            return True
    return False

def zjistiSegmentyTrid(rozdeleniDoTrid_k_means, listQ):
    segmentyTrid = []
    for trida in rozdeleniDoTrid_k_means:
        bod = trida[0] # beru prvni bod ve tride
        segmentyTrid.append(zjistiSegmentProBod(bod, listQ))
    return segmentyTrid


# Metoda konstantních přírůstků
def konstantniPrirustky(rozdeleni_Kmeans, q):
    listOmega = [1, -1]
    okList = []

    pomocnyListOmega = []
    listVektoru = []
    for i in range(len(rozdeleni_Kmeans)):
        trida = rozdeleni_Kmeans[i]
        for vektor in trida:
            listVektoru.append(vektor)
            pomocnyListOmega.append(listOmega[i])

    return konstantniPrirustky_pomocna(listVektoru, pomocnyListOmega, q)

def konstantniPrirustky_pomocna(listVektoru, pomocnyListOmega, q):
    for idx_konstPri in range(500000): # zamezeni nekonecneho cyklu
        kontrola = kontrolaOK(listVektoru, pomocnyListOmega, q)
        if (jsouVsechnyOk(kontrola)):
            print("Počet iterací - metoda konstantních přírůstků: {}".format(idx_konstPri + 1))
            return q
     
        mamNoveQ = False
        for i in range(len(listVektoru)):
            vektor = listVektoru[i]
            obrazovyVektor_rozsireny = [1]
            for j in range(len(vektor)):
                obrazovyVektor_rozsireny.append(vektor[j]) 
            obrazovyVektor_rozsireny = tuple(obrazovyVektor_rozsireny)
            soucin_Q_vektorRozsireny = vypoctiSoucinQ_vektorRozsireny(q, obrazovyVektor_rozsireny, pomocnyListOmega[i])
            if(soucin_Q_vektorRozsireny < 0):
                q = vypoctiNoveQ(q, obrazovyVektor_rozsireny, pomocnyListOmega[i], 0.1)
                mamNoveQ = True

        if(not mamNoveQ):
            print("Počet iterací - metoda konstantních přírůstků: {}".format(idx_konstPri + 1))
            return q
            
    print("Počet iterací - metoda konstantních přírůstků: {}".format(idx_konstPri + 1))
    return q

def ZjistiDeliciPrimkuKonstPrirustky(trida1, trida2, q0):
    data_konst_prirustky = []

    data_konst_prirustky.append(trida1)
    data_konst_prirustky.append(trida2)
    
    q_final = konstantniPrirustky(data_konst_prirustky, q0)
    body_Usecka = spocitejUsecku(q_final)
    return body_Usecka, q_final


# Upravená metoda konstantních přírůstků
def upraveneKonstantniPrirustky(rozdeleni_Kmeans, q):
    listOmega = [1, -1]
    okList = []

    pomocnyListOmega = []
    listVektoru = []
    for i in range(len(rozdeleni_Kmeans)):
        trida = rozdeleni_Kmeans[i]
        for vektor in trida:
            listVektoru.append(vektor)
            pomocnyListOmega.append(listOmega[i])

    return upraveneKonstantniPrirustky_pomocna(listVektoru, pomocnyListOmega, q)

def upraveneKonstantniPrirustky_pomocna(listVektoru, pomocnyListOmega, q):
    for idx_konstPriUpr in range(500000):
        kontrola = kontrolaOK(listVektoru, pomocnyListOmega, q)
        if (jsouVsechnyOk(kontrola)):
            print("Počet iterací - upravená metoda konstantních přírůstků: {}".format(idx_konstPriUpr + 1))
            return q

        mamNoveQ = False

        for i in range(len(listVektoru)):
            vektor = listVektoru[i]
            obrazovyVektor_rozsireny = [1]
            for j in range(len(vektor)):
                obrazovyVektor_rozsireny.append(vektor[j]) 
            obrazovyVektor_rozsireny = tuple(obrazovyVektor_rozsireny)
            soucin_Q_vektorRozsireny = vypoctiSoucinQ_vektorRozsireny(q, obrazovyVektor_rozsireny, pomocnyListOmega[i])
            while(soucin_Q_vektorRozsireny < 0):
                q = vypoctiNoveQ(q, obrazovyVektor_rozsireny, pomocnyListOmega[i], 0.1)
                mamNoveQ = True
                soucin_Q_vektorRozsireny = vypoctiSoucinQ_vektorRozsireny(q, obrazovyVektor_rozsireny, pomocnyListOmega[i])
        
        if(not mamNoveQ):
            print("Počet iterací - upravená metoda konstantních přírůstků: {}".format(idx_konstPriUpr + 1))
            return q
    print("Počet iterací - upravená metoda konstantních přírůstků: {}".format(idx_konstPriUpr + 1))
    return q

def ZjistiDeliciPrimkuKonstPrirustky_uprav(trida1, trida2, q0):
    data_konst_prirustky_uprav = []

    data_konst_prirustky_uprav.append(trida1)
    data_konst_prirustky_uprav.append(trida2)
    
    q_final = upraveneKonstantniPrirustky(data_konst_prirustky_uprav, q0)
    body_Usecka = spocitejUsecku(q_final)
    return body_Usecka, q_final


# METODY - Vykreslování grafů
def vykresliData(data):
    print("Vykresluji data...")
    for vektor in data:
        X = vektor[0]
        Y = vektor[1]
        plt.scatter(X, Y) 
        plt.title("Vykreslení dat")
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()

def vykresliRetezovouMapu(vektory_postupne, index_pocatek):
    bod_pocatek = data[index_pocatek]
    x_values = []
    y_values = []
    for vektor in vektory_postupne:
        x_values.append(vektor[0])
        y_values.append(vektor[1])
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Řetězová mapa (počáteční index: %i)" %index_pocatek)
    plt.plot(x_values, y_values, linestyle='solid',color='blue', marker = 'x')
    plt.scatter(bod_pocatek[0], bod_pocatek[1], s=200, color = 'black', marker = '+')
    plt.show()

def vykresli_k_means(rozdeleniDoTrid_k_means, stredy_kmeans):
    print("Vykresluji k-means rozdělení...")
    colours = ["red", "green", "yellow", "magenta"]
    i = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    for stred in stredy_kmeans:
        plt.scatter(stred[0], stred[1], color = 'black', marker = '+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("K-means")
    plt.show()

def vykresli_it_opt(iterativniOptimalizace):
    print("Vykresluji iterativní optimalizaci...")
    colours = ["red", "green", "yellow", "magenta"]
    i = 0
    for trida in iterativniOptimalizace:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i+= 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Iterativní optimalizace")
    plt.show()

def vykresli_nerov_deleni(rozdeleni_binarne_nerov):
    print("Vykresluji nerovnoměrné dělení...")
    colours = ["red", "green", "yellow", "magenta"]
    i = 0
    for trida in rozdeleni_binarne_nerov:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Binární dělení (nerovnoměrné)")
    plt.show()

def vykresli_rov_deleni(rozdeleni_binarne_rov):
    print("Vykresluji rovnoměrné dělení...")
    colours = ["red", "green", "yellow", "magenta"]
    i = 0
    for trida in rozdeleni_binarne_rov:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Binární dělení (rovnoměrné)")
    plt.show()

def zjistiMinMaxX(data):
    data_x_values = []
    
    for vektor in data:
        data_x_values.append(vektor[0]) 
      
    x_min = min(data_x_values)
    x_max = max(data_x_values)

    return (x_min, x_max)

def zjistiMinMaxY(data):
    data_y_values = []

    for vektor in data:
        data_y_values.append(vektor[1])

    y_min = min(data_y_values)
    y_max = max(data_y_values)

    return (y_min, y_max)

def vykresliBayes(stredy_z_k_means, rozdeleniDoTrid_k_means,  list_apr_ppst_tridy, list_kovariancniMatice, data):
    print("Vykresluji Bayesův klasifikátor...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0], X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            indexTridy = Bayesovsky_klasifikator(stredy_z_k_means, rozdeleniDoTrid_k_means, list_apr_ppst_tridy, list_kovariancniMatice, data, bod)
            plt.scatter(ix, jy, color = colours[indexTridy], marker = '+')
    i = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    plt.title("Bayesův klasifikátor")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def vykresli_vektorova_kvantizace(stredy_z_k_means, rozdeleniDoTrid_k_means, data):
    print("Vykresluji vektorovou kvantizaci...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0], X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            indexTridy = vektorovaKvantizace_Kmeans(bod, stredy_z_k_means, rozdeleniDoTrid_k_means)
            plt.scatter(ix, jy, color = colours[indexTridy], marker = '+')
    i = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    plt.title("Vektorová kvantizace")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def vykresliKnn(rozdeleniDoTrid_k_means, K, data):
    print("Vykresluji knn...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0],  X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            indexTridy = knn(rozdeleniDoTrid_k_means.copy(), bod, K)
            plt.scatter(ix, jy, color = colours[indexTridy], marker = '+')
    i = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[i])
        i += 1
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Nejbližší soused (K = %i)' %K)
    plt.show()

def vykresliRosenblatt(rozdeleniDoTrid_k_means, usecky_Ros, listQ_Ros, segmentyTrid, data):
    print("Vykresluji Rosenblattův algoritmus...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0], X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            segmentBodu = zjistiSegmentProBod(bod, listQ_Ros)
            if(patriBodDoTridy(segmentBodu, segmentyTrid)): 
                plt.scatter(ix, jy, color = colours[segmentyTrid.index(segmentBodu)], marker = '+')
            else:
                plt.scatter(ix, jy, color = "grey", marker = '+')

    colorIndex = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[colorIndex])
        colorIndex += 1
    
    for usecka in usecky_Ros:
        plt.plot(usecka[0], usecka[1], marker = 'o')
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(Y_min_max[0], Y_min_max[1])
    plt.title("Rosenblattův algoritmus")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def vykresliMetoduKonstPrirustku(rozdeleniDoTrid_k_means, usecky_KonstPrir, listQ_KonstPrir, segmentyTrid, data):
    print("Vykresluji metodu konstantních přírůstků...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0], X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            segmentBodu = zjistiSegmentProBod(bod, listQ_KonstPrir)
            if(patriBodDoTridy(segmentBodu, segmentyTrid)): 
                plt.scatter(ix, jy, color = colours[segmentyTrid.index(segmentBodu)], marker = '+')
            else:
                plt.scatter(ix, jy, color = "grey", marker = '+')

    colorIndex = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[colorIndex])
        colorIndex += 1
    
    #for usecka in usecky_KonstPrir:
        #plt.plot(usecka[0], usecka[1], marker = 'o')
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(Y_min_max[0], Y_min_max[1])
    plt.title("Metoda konstantních přírůstků")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def vykresliUpravenouMetoduKonstPrirustku(rozdeleniDoTrid_k_means, usecky_konstPrir_uprav, listQ_konstPrir_uprav, segmentyTrid, data):
    print("Vykresluji upravenou metodu konstantních přírůstků...")
    colours = ["red", "green", "yellow", "magenta"]
    X_min_max = zjistiMinMaxX(data)
    Y_min_max = zjistiMinMaxY(data)
    for ix in np.arange(X_min_max[0], X_min_max[1], 1):
        for jy in np.arange(Y_min_max[0], Y_min_max[1], 1):
            bod = (ix, jy)
            segmentBodu = zjistiSegmentProBod(bod, listQ_konstPrir_uprav)
            if(patriBodDoTridy(segmentBodu, segmentyTrid)): 
                plt.scatter(ix, jy, color = colours[segmentyTrid.index(segmentBodu)], marker = '+')
            else:
                plt.scatter(ix, jy, color = "grey", marker = '+')

    colorIndex = 0
    for trida in rozdeleniDoTrid_k_means:
        for vektor in trida:
            X = vektor[0]
            Y = vektor[1]
            plt.scatter(X, Y, color = colours[colorIndex])
        colorIndex += 1
    
    #for usecka in usecky:
        #plt.plot(usecka[0], usecka[1], marker = 'o')
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(Y_min_max[0], Y_min_max[1])

    plt.title("Upravená metoda konstantních přírůstků")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# VÝKONNÝ KÓD
if __name__ == "__main__":

    # Název souboru
    #filename = "data_2Tridy_Ros.txt"
    filename = "data.txt"
    #filename = "data-mala-2.txt"

    # Načítání dat
    data = load(filename)
    
    # Matice vzdálenosti
    #maticeVzdalenosti = vypoctiMaticiVzdalenosti(data)
    #print(maticeVzdalenosti)

    # URČENÍ POČTU TŘÍD
    #shluková hladina
    #pocet_trid_shlukovaHladina = shlukovaHladina(maticeVzdalenosti.copy())

    # Řetězová mapa - volaná 1x
    #index_pocatek_random = rand.randint(0, len(data)-1)
    #retezovaMapa_ = retezovaMapa(maticeVzdalenosti.copy(), index_pocatek_random)
    #print("Řetězová mapa (1 volání) - počet tříd je: %d" %(retezovaMapa_[0]))

    # Vykreslování řetězové mapy
    #vykresliRetezovouMapu(retezovaMapa_[1], index_pocatek_random)

    # Řetězová mapa - volaná n krát
    list_retezovaMapa = []
    n = 60
    #for i in range(n):
        #index_pocatek_random = rand.randint(0, len(data)-1)
        #pocetTrid_retez = retezovaMapa(maticeVzdalenosti.copy(), index_pocatek_random)
        #list_retezovaMapa.append(pocetTrid_retez[0])
    #retezovaMapa_pocetTrid = round(np.mean(list_retezovaMapa))
    #print("Řetězová mapa (%d volání) - počet tříd je: %d" %(n, retezovaMapa_pocetTrid))

    # MAXIMIN
    q = 0.5 
    #pocetTrid_MAXIMIN = MAXIMIN(maticeVzdalenosti.copy(), index_pocatek_random, q)

    # DĚLENÍ DO TŘÍD
    # K-means
    pocetTrid = 3 # potreba menit podle dat
    vysledek_k_means = k_means(pocetTrid, data)
    rozdeleniDoTrid_k_means = vysledek_k_means[0]
    stredy_z_k_means = vysledek_k_means[1]

    # Pro kontrolu
    vykresli_k_means(rozdeleniDoTrid_k_means.copy(), stredy_z_k_means)

    # Iterativní optimalizace
    #iterativniOptimalizace = iterativniOptimalizace(rozdeleniDoTrid_k_means, stredy_z_k_means.copy(), data)

    # Nerovnoměrné binární dělení do tříd 
    #binarniDeleni_nerov = binarniDeleni_nerovnomerne(maticeVzdalenosti.copy(), pocetTrid, data)
    #rozdeleni_binarne_nerov = binarniDeleni_nerov[0]
    #stredy_vekt_kvantizace_nerov = binarniDeleni_nerov[1]

    # KLASIFIKÁTORY
    # Bayesovský klasifikátor
    list_apr_ppst_tridy = vypoctiApriorniPravdepododnostiTrid(rozdeleniDoTrid_k_means, data)
    list_kovariancniMatice = vypoctiKovariancniMatice(rozdeleniDoTrid_k_means, stredy_z_k_means)

    # Klasifikátor podle jednoho a dvou nejbližších sousedů (nastaveni parametru K)
    K = 2

    # Vektorová kvantizace
    # viz vykreslování níže

    # Rosenblattův algoritmus
    usecky_Ros = []
    listQ_Ros = []
    q0 = (3, 2, 1)

    if(pocetTrid == 2):
        # Pro 2 třídy
        usecka1 = ZjistiDeliciPrimkuRosenblatt( rozdeleniDoTrid_k_means[1],  rozdeleniDoTrid_k_means[0], q0)
        usecky_Ros.append(usecka1[0])
        listQ_Ros.append(usecka1[1])

        segmentyTrid = zjistiSegmentyTrid(rozdeleniDoTrid_k_means, listQ_Ros)
        vykresliRosenblatt(rozdeleniDoTrid_k_means, usecky_Ros, listQ_Ros, segmentyTrid, data)

    # Pro 3 třídy
    usecka1 = ZjistiDeliciPrimkuRosenblatt(rozdeleniDoTrid_k_means[0], SpojTridy(rozdeleniDoTrid_k_means[1], rozdeleniDoTrid_k_means[2]), q0)
    usecky_Ros.append(usecka1[0])
    listQ_Ros.append(usecka1[1])

    usecka2 = ZjistiDeliciPrimkuRosenblatt(rozdeleniDoTrid_k_means[1], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[2]), q0)
    usecky_Ros.append(usecka2[0])
    listQ_Ros.append(usecka2[1])

    usecka3 = ZjistiDeliciPrimkuRosenblatt(rozdeleniDoTrid_k_means[2], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[1]), q0)
    usecky_Ros.append(usecka3[0])
    listQ_Ros.append(usecka3[1])

    segmentyTrid = zjistiSegmentyTrid(rozdeleniDoTrid_k_means, listQ_Ros)
    vykresliRosenblatt(rozdeleniDoTrid_k_means, usecky_Ros, listQ_Ros, segmentyTrid, data)

    # Metoda konstantních přírůstků
    usecky_KonstPrir = []
    listQ_KonstPrir = []
    q0 = (3, 2, 1)

    usecka1 = ZjistiDeliciPrimkuKonstPrirustky(rozdeleniDoTrid_k_means[0], SpojTridy(rozdeleniDoTrid_k_means[1], rozdeleniDoTrid_k_means[2]), q0)
    usecky_KonstPrir.append(usecka1[0])
    listQ_KonstPrir.append(usecka1[1])

    usecka2 = ZjistiDeliciPrimkuKonstPrirustky(rozdeleniDoTrid_k_means[1], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[2]), q0)
    usecky_KonstPrir.append(usecka2[0])
    listQ_KonstPrir.append(usecka2[1])

    usecka3 = ZjistiDeliciPrimkuKonstPrirustky(rozdeleniDoTrid_k_means[2], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[1]), q0)
    usecky_KonstPrir.append(usecka3[0])
    listQ_KonstPrir.append(usecka3[1])

    # Upravená metoda konstantních přírůstků
    usecky_konstPrir_uprav = []
    listQ_konstPrir_uprav = []
    q0 = (3, 2, 1)

    usecka1 = ZjistiDeliciPrimkuKonstPrirustky_uprav(rozdeleniDoTrid_k_means[0], SpojTridy(rozdeleniDoTrid_k_means[1], rozdeleniDoTrid_k_means[2]), q0)
    usecky_konstPrir_uprav.append(usecka1[0])
    listQ_konstPrir_uprav.append(usecka1[1])

    usecka2 = ZjistiDeliciPrimkuKonstPrirustky_uprav(rozdeleniDoTrid_k_means[1], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[2]), q0)
    usecky_konstPrir_uprav.append(usecka2[0])
    listQ_konstPrir_uprav.append(usecka2[1])

    usecka3 = ZjistiDeliciPrimkuKonstPrirustky_uprav(rozdeleniDoTrid_k_means[2], SpojTridy(rozdeleniDoTrid_k_means[0], rozdeleniDoTrid_k_means[1]), q0)
    usecky_konstPrir_uprav.append(usecka3[0])
    listQ_konstPrir_uprav.append(usecka3[1])


    # VYKRESLOVÁNÍ GRAFŮ
    # Vykreslování dat
    vykresliData(data)

    # Vykreslování k-means
    vykresli_k_means(rozdeleniDoTrid_k_means.copy(), stredy_z_k_means)

    # Vykreslování iterativní optimalizace
    vykresli_it_opt(iterativniOptimalizace)

    # Vykreslování nerovnoměrného binárního dělení
    vykresli_nerov_deleni(rozdeleni_binarne_nerov)

    # Vykreslování Bayesova klasifikátoru
    vykresliBayes(stredy_z_k_means, rozdeleniDoTrid_k_means,  list_apr_ppst_tridy, list_kovariancniMatice, data)

    # Vykreslování vektorove kvantizace
    vykresli_vektorova_kvantizace(stredy_z_k_means, rozdeleniDoTrid_k_means.copy(), data)

    # Vykreslování nejbližšího souseda
    vykresliKnn(rozdeleniDoTrid_k_means, K, data)

    # Vykreslování Rosenblattova algoritmu
    vykresliRosenblatt(rozdeleniDoTrid_k_means, usecky_Ros, listQ_Ros, segmentyTrid, data)

    # Vykreslení metody konstantních přírustků
    vykresliMetoduKonstPrirustku(rozdeleniDoTrid_k_means, usecky_KonstPrir, listQ_KonstPrir, segmentyTrid, data)

    # Vykreslení upravené metody konstantních přírůstků
    vykresliUpravenouMetoduKonstPrirustku(rozdeleniDoTrid_k_means, usecky_konstPrir_uprav, listQ_konstPrir_uprav, segmentyTrid, data)




    
    








