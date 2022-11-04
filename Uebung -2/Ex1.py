print ('Uebung 2 Aufgabe 1')
from sklearn.metrics.pairwise import cosine_similarity

print ('######################a###################')
A = [3.06, 500, 6]
B = [2.68, 320, 4]
C = [2.92, 640, 6]

cosAB = cosine_similarity([A], [B])
cosAC = cosine_similarity([A], [C])
cosBC = cosine_similarity([B], [C])

print("Angle between A and B:", cosAB[0][0])
print("Angle between A and C:", cosAC[0][0])
print("Angle between B and C:", cosBC[0][0])


print ('######################b#####################')
A = [3.06*1, 500*0.01, 6*0.5]
B = [2.68*1, 320*0.01, 4*0.5]
C = [2.92*1, 640*0.01, 6*0.5]

cosAB = cosine_similarity([A], [B])
cosAC = cosine_similarity([A], [C])
cosBC = cosine_similarity([B], [C])

print("Angle between A and B:", cosAB[0][0])
print("Angle between A and C:", cosAC[0][0])
print("Angle between B and C:", cosBC[0][0])

print ('#########################c##############################')
import statistics

disk = [500, 320, 640]
memory = [6, 4, 6]

alpha = 1/statistics.mean(disk)
beta = 1/statistics.mean(memory)

A = [3.06, 500*alpha, 6*beta]
B = [2.68, 320*alpha, 4*beta]
C = [2.92, 640*alpha, 6*beta]

cosAB = cosine_similarity([A], [B])
cosAC = cosine_similarity([A], [C])
cosBC = cosine_similarity([B], [C])

print("Alpha:", alpha)
print("Beta:", beta)
print("Angle between A and B:", cosAB[0][0])
print("Angle between A and C:", cosAC[0][0])
print("Angle between B and C:", cosBC[0][0])