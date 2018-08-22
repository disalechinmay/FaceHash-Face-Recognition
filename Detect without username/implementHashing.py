import os

def findHash(ratios):
	customHash = ""
	hashChars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	for x in ratios:
		tempx = x / 1
		#print(tempx)
		baseNumber = str(tempx).split('.')[0]
		customHash += hashChars[int(baseNumber)]
		'''
		decimalPortion = tempx % 1
		if decimalPortion < 0.5:
			customHash += '0'
		else:
			customHash += '1'
		'''
	return customHash

#Read dataStore.txt
lines = [line.rstrip('\n') for line in open('dataStore.txt')]
for line in lines:
	#print(line)

	#line = "Chinmay 6.45843718125 7.41951179536 8.52366720277 8.88807609962 1.0 2.97353873096 4.03975331331 4.05508822806 4.10536866385 1.78698678034 3.43341004581 2.6568524898 1.54769042034 3.88050132487 3.19797104976 2.40946409685 5.3270001584 3.87426678242 1.78404967602 4.97669359625 3.68787662357 1.69494358266 "

	os.chdir("/home/chinmay/MASTERS/SEM1/STANDALONE FINAL/Recog Tuned/Hashes")

	ratios = line.split()
	userName = ratios[0]
	finalRatios = []
	for x in range(1, len(ratios)) :
		finalRatios.append(float(ratios[x]))

	computedHash = findHash(finalRatios)
	print(computedHash)

	splitHash = [computedHash[i:i+1] for i in range(0, len(computedHash), 2)]

	print(splitHash)

	path = ""
	for x in splitHash:
		path += x
		path += "/"
	masterPath = "/home/chinmay/MASTERS/SEM1/STANDALONE FINAL/Recog Tuned/Hashes/" + path
	#print(path)

	if os.path.exists(path):
		print("Path already exists!")
	else:
		os.makedirs(path)

	os.chdir(path)
	fileHandler = open(userName, "w+")
	fileHandler.write("%s" % line)
	fileHandler.close()

	#print(os.listdir(masterPath))

	os.chdir("/home/chinmay/MASTERS/SEM1/STANDALONE FINAL/Recog Tuned/Hashes")