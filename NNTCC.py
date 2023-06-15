from sklearn.neighbors import KDTree
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches
from matplotlib.patches import Circle
from sympy import *
from heapq import nsmallest
from collections import namedtuple
from collections import Counter
from sympy.plotting import plot
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.multiclass import OneVsRestClassifier


def find_second_smallest(a: list) -> int:
    f1, f2 = float('inf'), float('inf')
    for i in range(len(a)):
        if a[i] <= f1:
            f1, f2 = a[i], f1
        elif a[i] < f2 and a[i] != f1:
            f2 = a[i]
    return f2


def r_tangent_circle(a, b, c):
    A = math.sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2)
    B = math.sqrt((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2)
    C = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    Ra = (B + C - A) / 2
    Rb = (C + A - B) / 2
    Rc = (A + B - C) / 2
    return [Ra, Rb, Rc]


#Circle = namedtuple('Circle', 'x, y, r')

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


""" STEP ONE """
# load data
with open('jain.txt') as f:
    points = [tuple(map(float, i.split('\t')[0:2])) for i in f]

#NaN_Searching
tree = KDTree(points)
num_list = [1000, 1001]
r = 1
landa = len(points)

while r < landa:

    num = 0
    all_nn_indices = tree.query_radius(points, r)
    all_nns = [[idx for idx in nn_indices if idx != i] for i, nn_indices in enumerate(all_nn_indices)]

    for i, nns in enumerate(all_nns):

        nns_set = set(nns)
        rnns = [all_nns.index(nns) for nns in all_nns if i in nns]

        for Nb in range(len(rnns) == 0):
            num = num + 1
    num_list.append(num)

    if num_list[r] == num_list[r - 1]:

            break
    else:
        r = r + 1
        continue

num_list.remove(1000)
num_list.remove(1001)
landa = r-1
print("landa =", landa)
print(num_list)


def nn(input):
    r = input
    all_nn_indices = tree.query_radius(points, r)
    all_nns = [[idx for idx in nn_indices if idx != i] for i, nn_indices in enumerate(all_nn_indices)]
    rnns = [[all_nns.index(nns) for nns in all_nns if i in nns] for i, nns in enumerate(all_nns)]

    return rnns

n = 0
TT_NaN_list = []
for i, m in enumerate(nn(landa)):
    print(n)
    n = n + 1
    rnns_set = set(m)
    total_nan = rnns_set
    print("T_NaN = ", list(total_nan))
    print("Nb = ", len(m))
    TT_NaN_list.insert(i, m)


# finding two nearest neighbor of each point

nearest_NaN_sets=[]   #list of sets

prdes=[]  # contains points and source of each points radiuses

for j, tt in enumerate(TT_NaN_list):
    tt1 = list(tt)
    d = []
    prdes.append([j])
    for t in tt1:
        d.append(distance(points[j], points[t]))

    if len(tt) >=2:
        m1 = min(d)
        m2 = find_second_smallest(d)
        if m1 == m2:
            repeated = m2
            indx = [i for i in range(len(d)) if d[i] == repeated]
            set1 = {j, tt1[indx[0]], tt1[indx[1]]}
            nearest_NaN_sets.append(set1)
        else:
            set1 = {j, tt1[d.index(m1)], tt1[d.index(m2)]}
            nearest_NaN_sets.append(set1)

nearest_NaN_lists=[]    # lists of each point's two nearest neighbors
for s1, sets in enumerate(nearest_NaN_sets):

    nearest_NaN_lists.append(list(sets))


# computing radiuses
circles = []
for h, d4 in enumerate(nearest_NaN_lists):
 if len(d4)==3:
    r3 = r_tangent_circle(points[d4[0]], points[d4[1]], points[d4[2]])
    r4 = (r3[0] + r3[1] + r3[2]) / 3

    if r3[0] and r3[1] and r3[2] != 0:
        r01 = r3[0] / r3[1]
        r02 = r3[0] / r3[2]
        r10 = r3[1] / r3[0]
        r12 = r3[1] / r3[2]
        r20 = r3[2] / r3[0]
        r21 = r3[2] / r3[1]

        if r01 < 0.46 or r02 < 0.46:
            prdes[d4[0]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[0]].append([r3[0]] + [nearest_NaN_lists.index(d4)])

        if r10 < 0.46 or r12 < 0.46:
            prdes[d4[1]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[1]].append([r3[1]] + [nearest_NaN_lists.index(d4)])

        if r20 < 0.46 or r21 < 0.46:
            prdes[d4[2]].append([r4] + [nearest_NaN_lists.index(d4)])
        else:
            prdes[d4[2]].append([r3[2]] + [nearest_NaN_lists.index(d4)])

qq = 0
rs = [] # contains minimum radius of points

not_TT_radius=[]

for de, dus in enumerate(prdes):
    if len(dus)==1:
        not_TT_radius.append(dus[0])

    ttt = TT_NaN_list[dus[0]]
    qq = qq + len(dus)
    les = []

    le = len(dus)
    if le >= 1:
        for ee in range(1, le):
            les.append(dus[ee][0])

    if les == []:
        ra = 0
    else:
        ra = min(les)
        rs.append((de, ra))

    dus.remove(dus[0])
    circle03 = plt.Circle(points[de], ra, color='orange', fill=False)
    plt.gcf().gca().add_artist(circle03)
    circles.append(circle03)


""" STEP TWO """
srs = sorted(rs, key=lambda tup: tup[1])  # sort points by radius
#print("sort by radius", srs)


srs_id=[]     # only sorted points indexes
for to, top in enumerate(srs):
    srs_id.append(top[0])


deleted=[]
for to, top in enumerate(srs_id):

     set1 = set(TT_NaN_list[top])
     if to < len(srs_id):
      for tos in range(0, to):

         if srs_id[tos] not in deleted:
            sb = TT_NaN_list[srs_id[tos]]
            disb1 = set1.intersection(set(sb))

            if disb1 != set():

             deleted.append(top)


remained = list(set(srs_id) - set(deleted))



minm = [] # min distances for remained points

for mo in remained:
    mm = []
    mmd = []
    for mo2 in remained:
        if mo != mo2:
            mm.append(distance(points[mo], points[mo2]))
            mmd.append((distance(points[mo], points[mo2]),) + (mo2,))
    minm.append((min(mm),))



mdis = [] # a list of remained tuples in shape: (radius, max(min_distance), point_id)
mdis1 = []
for mj, mo in enumerate(remained):
    for cr, tpp in enumerate(srs):
        if tpp[0] == mo:
            mdis.append((tpp[1],) + minm[mj] + (tpp[0],))
            mdis1.append((tpp[1],) + minm[mj])


smdis = sorted(mdis, key=lambda tup: tup[0]) #sorted remained points in ascending order by radius
smdis2 = sorted(mdis, key=lambda tup: tup[1], reverse=True)#sorted remained points in discending order by max(min_distance)



""" STEP THREE """
# computing threshold
L = len(smdis)
rn = ((smdis[L - 1][0] - smdis[0][0]) / 2) + smdis[0][0]
print(rn)
if smdis2[0][0] and smdis2[1][0] < rn:
    dis = distance(points[smdis2[0][2]], points[smdis2[1][2]]) / 2
    print(dis)
else:
    dis = distance(points[smdis[0][2]], points[smdis[1][2]]) / 2
    print(dis)


sort_remained = []
centers = [smdis[0][2]]
for m22, smd in enumerate(smdis):
    sort_remained.append(smd[2])

    dises = []
    u = 0
    for ts in range(0, m22):
        dist2 = distance(points[smdis[m22][2]], points[smdis[ts][2]]) / 2
        dises.append(dist2)
        if dises[ts] >= dis:
            u = u + 1
        if u == m22:
            if int(smdis[m22][0] / smdis[0][0]) == 1:
                centers.append(smdis[m22][2])

print("sort_remained", sort_remained)
print(centers)
sort_centers=sorted(centers)
print(sort_centers)


""" STEP FOUR """
#clustering
cens = []
xcens = []
ycens = []

allocate=[] #assigned points to clusters
xcentrs=[]
ycentrs=[]
for ci, cc in enumerate(sort_centers):
    allocate.append(cc)
    cens.append([cc])
    xcens.append([points[cc][0]])
    ycens.append([points[cc][1]])
    xcentrs.append([points[cc][0]])
    ycentrs.append([points[cc][1]])
halo = [] #not allocated points
xhalo = []
yhalo = []


for point in points:
    if points.index(point) not in allocate:
        halo.append(points.index(point))
        xhalo.append(point[0])
        yhalo.append(point[1])

    for cs in range(0, len(centers)):

        if points.index(point) in TT_NaN_list[cens[cs][0]]:
            allocate.append(points.index(point))
            cens[cs].append(points.index(point))
            xcens[cs].append(point[0])
            ycens[cs].append(point[1])

print("cens",cens)


if len(allocate)!=len(points):
 while len(halo)!=0:
  for hs in halo:
     mdh = []
     for nh in allocate:
       mdh.append(distance(points[hs], points[nh]))
     for ps in cens:
        if set(TT_NaN_list[hs]).intersection(set(ps)) != set():
          if allocate[mdh.index(min(mdh))] in ps:

              if hs not in allocate:
                allocate.append(hs)
              halo.remove(hs)
              ps.append(hs)
              xcens[cens.index(ps)].append(points[hs][0])
              ycens[cens.index(ps)].append(points[hs][1])




"""ploting"""
plt.scatter(*zip(*points), s=0.1)
for i, txt in enumerate(points):
    plt.annotate(i, points[i])
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('jain1.tif')
plt.show()

plt.scatter(*zip(*mdis1))
for i, txt in enumerate(remained):
    plt.annotate(remained[i], mdis1[i])
plt.xlabel('radius')
plt.ylabel('min distance')
plt.savefig('jain2.tif')
plt.show()


tplecenter=[]
stcens=[]
for kh in range(0,len(centers)):
    tplecenter.append(points[centers[kh]])
    stcens.append((sort_centers[kh],))

sizes=[40]

plt.figure(dpi=150)
styles = ['lightgreen', 'salmon', 'lightblue', 'purple', 'b', 'm', 'y', '#9400D3', '#C0FF3E']
plt.scatter(xhalo, yhalo, color='black', s=10)
for ls in range(0, len(centers)):
    plt.scatter(xcens[ls], ycens[ls], color=styles[ls], s=10, label='cluster'+str(ls+1))
    plt.scatter(xcentrs[ls], ycentrs[ls], color='black', s=20)
    labels = stcens
    for xc1, yc1, label, size in zip(xcentrs[ls], ycentrs[ls], labels[ls], sizes):
        plt.annotate(label, (xc1, yc1), fontsize=20)


plt.scatter(xcentrs[0], ycentrs[0], color='black', s=10, label='cluster heads')

plt.legend(loc='upper right', fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('jain3.tif')
plt.show()




""" analyzing """

with open('jain.txt') as f:
    y1=[ int(i.split('\t')[2]) for i in f]

true_target=y1   # true lables

mytarget1=[] # my predicted lable
for point in points:
    for ps in cens:
        if points.index(point) in ps:
            mytarget1.append(cens.index(ps))



"""aggregation data set label normlizing"""
"""
mytarget=[]
for tr in mytarget1:
    if tr==0:
        mytarget.append(2)
    if tr==1:
        mytarget.append(7)
    if tr==2:
        mytarget.append(4)
    if tr==3:
        mytarget.append(3)
    if tr==4:
        mytarget.append(6)
    if tr ==5:
        mytarget.append(1)
    if tr==6:
        mytarget.append(5)
"""


"""flame lable normlizing"""
"""
mytarget=[]
for tr in mytarget1:
    if tr==0:
        mytarget.append(1)
    if tr==1:
        mytarget.append(2)
    if tr==2:
        mytarget.append(3)
"""
"""jain lable normlizing"""

mytarget=[]
for tr in mytarget1:
    if tr==1:
        mytarget.append(1)
    if tr==0:
        mytarget.append(2)

"""spiral lable normlizing"""
"""
mytarget=[]
for tr in mytarget1:
    if tr==1:
        mytarget.append(1)
    if tr==2:
        mytarget.append(2)
    if tr==0:
        mytarget.append(3)
"""
X = np.array(points)
y = np.array(true_target)
# Use label_binarize to be multi-label like settings
Y = label_binarize(y, classes=[0, 1,2])
n_classes = Y.shape[1]

random_state = np.random.RandomState(0)
# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,random_state=random_state)


# We use OneVsRestClassifier for multi-label prediction
# Run classifier
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)


# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

y_true = true_target
y_pred = mytarget
multilabel_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("P",precision_score(y_true, y_pred, average='macro'))
print("R",recall_score(y_true, y_pred, average='macro'))
print("A",accuracy_score(y_true, y_pred))
print("F1",f1_score(y_true, y_pred, average='macro'))


print ("AMI",adjusted_mutual_info_score(y_true,y_pred, average_method='max'))
print ("ARI",adjusted_rand_score(y_true,y_pred))
print ("FMI",fowlkes_mallows_score(y_true,y_pred))
print ("NMI",normalized_mutual_info_score(y_true,y_pred,average_method='max'))
