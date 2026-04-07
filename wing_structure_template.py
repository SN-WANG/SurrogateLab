#!/usr/bin/env python
# -*- coding:utf-8 -*-
# coding=uf-8
# a=Ψ?
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

executeOnCaeStartup()
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import numpy as np
from odbAccess import openOdb
# >>> 修改点1：引入必要的库 <<<
import os
import time

openMdb(pathName='wing_structure_model.cae')
###############################################################################
###############              定义设计变量：不同分区的蒙皮厚度              ##############################
thick1=6.0529
thick2=7.3003
thick3=5.1129
###############################################################################
# 定义载荷点，1为最前一个参考点，3为最后一个
loadCases = 3
# 各个分区内的加强数量
num = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# num=[2,2,2,2,2,2,2,2,2]
# 网格尺寸
meshSize = 30

# 定义建模过程的定格时间
sleepTime = 0.001

# reSketchOrigin(-800,0,0,0,0,0,0)
# reSketchOrigin(0,-800,0,0,0,0,0)
# reSketchOrigin(0,0,-600,0,0,0,0)
# reSketchOrigin(0,0,0,-400,0,0,0)
# reSketchOrigin(0,0,0,0,300,0,0)
# reSketchOrigin(0,0,0,0,0,-600,0)
mdb.models['Model-1'].sections['Section-2'].setValues(preIntegrate=OFF,
                                                      material='Material-1', thicknessType=UNIFORM, thickness=thick1,
                                                      thicknessField='', idealization=NO_IDEALIZATION,
                                                      integrationRule=SIMPSON,
                                                      numIntPts=5)
mdb.models['Model-1'].sections['Section-3'].setValues(preIntegrate=OFF,
                                                      material='Material-1', thicknessType=UNIFORM, thickness=thick2,
                                                      thicknessField='', idealization=NO_IDEALIZATION,
                                                      integrationRule=SIMPSON,
                                                      numIntPts=5)
mdb.models['Model-1'].sections['Section-5'].setValues(preIntegrate=OFF,
                                                      material='Material-1', thicknessType=UNIFORM, thickness=thick3,
                                                      thicknessField='', idealization=NO_IDEALIZATION,
                                                      integrationRule=SIMPSON,
                                                      numIntPts=5)
position1 = [5000, 3000, 1500, 0, -1500, -3000, -5000]
position2 = [-2500, -318.18181818199, 1318.18181818173, 3000]

p = mdb.models['Model-1'].parts['Part-1']

s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                             sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.Line(point1=(-41.25, 22.5), point2=(-8.75, -2.5))
mdb.models['Model-1'].sketches.changeKey(fromName='__profile__',
                                         toName='Sketch-8')
s1.unsetPrimaryObject()
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                            sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
TrimLine = s.Line(point1=(-2500.0, 5000.0), point2=(3000.0, -41.6666666666702))

# 横向墙
for j in range(6):
    if num[j] == 0:
        continue

    for i in range(num[j]):
        print(11111)
        StringerLine = s.Line(
            point1=(-3000.0, (i + 1) * (position1[j] - position1[j + 1]) / (num[j] + 1) + position1[j + 1]),
            point2=(4000, (i + 1) * (position1[j] - position1[j + 1]) / (num[j] + 1) + position1[j + 1]))
    # s.autoTrimCurve(curve1=StringerLine, point1=(3900, (i+1)*(position1[j]-position1[j+1])/3.0+position1[j+1]))

# 纵向肋
for j in range(6, 9):
    # print(num[j])
    pos = j - 6
    if num[j] == 0:
        continue

    for i in range(num[j]):
        # print(i)
        StringerLine = s.Line(
            point1=((-position2[pos] + position2[pos + 1]) / (num[j] + 1) * (i + 1) + position2[pos], 5000),
            point2=((-position2[pos] + position2[pos + 1]) / (num[j] + 1) * (i + 1) + position2[pos], -5000))

s.delete(objectList=(TrimLine,))
# 草图显示#################################################################
# session.viewports['Viewport: 1'].setValues(displayedObject=p)
# session.viewports['Viewport: 1'].setValues(displayedObject=None)
# g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
# s.setPrimaryObject(option=STANDALONE)
# time.sleep(sleepTime)
##################################################################
mdb.models['Model-1'].sketches.changeKey(fromName='__profile__',
                                         toName='Sketch-test')
s1.unsetPrimaryObject()

# 拉伸筋条

axis = p.DatumAxisByPrincipalAxis(principalAxis=YAXIS)
datum = p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=0.0)
mdb.models['Model-1'].parts['Part-1'].features.changeKey(
    fromName='Datum axis-1', toName='delete1')
d2 = p.datums

##################################################################################
# session.viewports['Viewport: 1'].setValues(displayedObject=p)
# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
# time.sleep(sleepTime)
#######################################################################################
# 初始骨架
t = p.MakeSketchTransform(sketchPlane=d2[datum.id], sketchUpEdge=d2[axis.id],
                          sketchPlaneSide=SIDE1, sketchOrientation=LEFT, origin=(1500.0, 0.0, 0.0))
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                             sheetSize=27394.04, gridSpacing=684.85, transform=t)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
s1.retrieveSketch(sketch=mdb.models['Model-1'].sketches['Sketch-origin'])
d1 = p.datums
p.ShellExtrude(sketchPlane=d2[datum.id], sketchUpEdge=d2[axis.id], sketchPlaneSide=SIDE1,
               sketchOrientation=LEFT, sketch=s1, depth=2000.0, flipExtrudeDirection=OFF)
s1.setPrimaryObject(option=STANDALONE)
time.sleep(sleepTime)
s1.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
temp = num.count(0)
##################################################################################
# session.viewports['Viewport: 1'].setValues(displayedObject=p)
# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
# time.sleep(sleepTime)
#######################################################################################
# 无加筋时候的备份
#
if temp != 9:
    t = p.MakeSketchTransform(sketchPlane=d2[datum.id], sketchUpEdge=d2[axis.id],
                              sketchPlaneSide=SIDE1, sketchOrientation=LEFT, origin=(1500.0, 0.0, 0.0))
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                 sheetSize=27394.04, gridSpacing=684.85, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.retrieveSketch(sketch=mdb.models['Model-1'].sketches['Sketch-test'])
    d1 = p.datums
    s1.setPrimaryObject(option=STANDALONE)
    time.sleep(sleepTime)
    p.ShellExtrude(sketchPlane=d2[datum.id], sketchUpEdge=d2[axis.id], sketchPlaneSide=SIDE1,
                   sketchOrientation=LEFT, sketch=s1, depth=2000.0, flipExtrudeDirection=OFF)
    time.sleep(sleepTime)
    s1.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']
# mdb.models['Model-1'].parts['Part-1'].sectionAssignments[1].resume()
else:
    # mdb.models['Model-1'].parts['Part-1'].sectionAssignments[2].suppress()
    # mdb.models['Model-1'].parts['Part-1'].sectionAssignments[1].setValues(
    # sectionName='Section-10')
    print('no reinforcedStiff')

##################################################################################
# session.viewports['Viewport: 1'].setValues(displayedObject=p)
# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
# time.sleep(sleepTime)
#######################################################################################
# 删除多余面
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
e = p.edges
edge = e.getByBoundingBox(-1600, -6000, 1900, 10000, 6000, 2000)
coordlist = []
for i in range(len(edge)):
    coord = edge[i].pointOn
    coordlist.append(coord)

face_rm = f[0:0]
for i in range(len(coordlist)):
    face = f.findAt(coordlist[i])
    if face:
        face_rm += f[face[0].index:face[0].index + 1]

p.RemoveFaces(faceList=face_rm, deleteCells=False)
time.sleep(sleepTime)
##################################################################################
# session.viewports['Viewport: 1'].setValues(displayedObject=p)
# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
# time.sleep(sleepTime)
#######################################################################################


# 建立集合
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
outterFaces = f[0:0]
reinforcedStiff = f[0:0]
originStiff = f[0:0]
bdSurfaces = f[0:0]
for i in p.faces:
    feature = i.featureName
    if (feature == 'Shell planar-1') or (feature == 'Shell planar-2') or (feature == 'Shell sweep-1'):
        outterFaces = outterFaces + f[i.index:i.index + 1]

p.Set(faces=outterFaces, name='allOutterFaces')

for i in p.faces:
    feature = i.featureName
    if (feature == 'Shell extrude-2'):
        reinforcedStiff = reinforcedStiff + f[i.index:i.index + 1]

p.Set(faces=reinforcedStiff, name='allreinforcedStiff')

for i in p.faces:
    feature = i.featureName
    if (feature == 'Shell extrude-1'):
        originStiff = originStiff + f[i.index:i.index + 1]

p.Set(faces=originStiff, name='alloriginStiff')

for i in p.faces:
    feature = i.featureName
    if (feature == 'Shell planar-3'):
        bdSurfaces = bdSurfaces + f[i.index:i.index + 1]

p.Set(faces=bdSurfaces, name='bdSurfaces')

# 装配
a = mdb.models['Model-1'].rootAssembly
a.regenerate()
a.Instance(name='Part-1-1', part=p, dependent=ON)


def defineStringer():
    # 初始化
    # for i in range(len(mdb.models['Model-1'].parts['Part-1'].stringers)):
    # stringersName=mdb.models['Model-1'].parts['Part-1'].stringers.keys()
    # del mdb.models['Model-1'].parts['Part-1'].stringers[stringersName[-1]]
    # p = mdb.models['Model-1'].parts['Part-1']
    # for num in range(len(mdb.models['Model-1'].parts['Part-1'].sectionAssignments)):
    # del mdb.models['Model-1'].parts['Part-1'].sectionAssignments[-1]
    count = 0
    allStringer = p.edges[0:0]
    topStringer = p.edges[0:0]
    for i in (p.sets['allLongStiff'].faces + p.sets['allTranStiff'].faces):
        for j in i.getEdges():
            nodeNum1 = p.edges[j].getVertices()[0]
            nodeNum2 = p.edges[j].getVertices()[1]
            node1 = p.vertices[nodeNum1].pointOn[0]
            node2 = p.vertices[nodeNum2].pointOn[0]
            length = sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) ** 2)
            if abs((node1[2] - node2[2]) / length) < 0.1:
                count = count + 1
                allStringer = allStringer + p.edges[j:j + 1]
                setName = 'Stringer' + str(count)
                # p.Set(edges=p.edges[j:j+1], name=setName)
                p.Stringer(edges=p.edges[j:j + 1], name=setName)
                p.Set(stringerEdges=(('Stringer%d' % count, p.edges[j:j + 1]),), name=setName)
                elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
                p.setElementType(regions=p.sets[setName], elemTypes=(elemType1,))
                if (node1[1] > 199 or node2[1] > 199):
                    topStringer = topStringer + p.edges[j:j + 1]
                p.SectionAssignment(region=p.sets[setName], sectionName='Section-Stringer', offset=0.0,
                                    offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
                p.assignBeamSectionOrientation(region=p.sets[setName], method=N1_COSINES, n1=(0.0, 0.0, -1.0))

    p.Set(edges=allStringer, name='allStringer')
    # p.Set(stringerEdges=((topStringer), ), name='topStringer')
    p.Set(edges=topStringer, name='topStringer')


# p.SectionAssignment(region=p.sets['allStringer'], sectionName='Section-Stringer', offset=0.0,
# offsetType=MIDDLE_SURFACE, offsetField='',thicknessAssignment=FROM_SECTION)


def defineSet():
    # setAll=[]
    # for i in mdb.models['Model-1'].parts['Part-1'].sets.items():
    # setAll.append(i[0])

    # setAll=tuple(setAll)
    # mdb.models['Model-1'].parts['Part-1'].deleteSets(setNames=setAll)
    p = mdb.models['Model-1'].parts['Part-1']
    f = p.faces
    longStiff = p.faces[0:0]
    tranStiff = p.faces[0:0]
    outterFaces = p.faces[0:0]
    longNum = 0
    transNum = 0
    outterNum = 0
    jishenFace = p.faces[0:0]
    f = p.faces
    faces = f.getByBoundingBox(-1001, -5001, -1, 4600, 5001, 201)
    # leadingEdge=f.getByBoundingCylinder((18.422034117E+03-0.1,-120.,-3.109731753E+03),(25.03824E+03+10,-120.,-6.9296E+03-10),10)
    for i in faces:
        temp = 0
        # if i in leadingEdge:
        # outterFaces=outterFaces+p.faces[i.index:i.index+1]
        # outterNum=outterNum+1
        # p.Set(faces=p.faces[i.index:i.index+1], name='outterStiff-%d'%outterNum)
        # continue
        normal = i.getNormal()
        if abs(normal[2]) > 0.85:
            outterFaces = outterFaces + p.faces[i.index:i.index + 1]
            outterNum = outterNum + 1
            p.Set(faces=p.faces[i.index:i.index + 1], name='outterStiff-%d' % outterNum)
            continue
        if abs(normal[1]) > 0.95:
            tranStiff = tranStiff + p.faces[i.index:i.index + 1]
            transNum = transNum + 1
            p.Set(faces=p.faces[i.index:i.index + 1], name='tranStiff-%d' % transNum)
            continue
        elif abs(normal[0]) > 0.85:
            longStiff = longStiff + p.faces[i.index:i.index + 1]
            longNum = longNum + 1
            p.Set(faces=p.faces[i.index:i.index + 1], name='longStiff-%d' % longNum)
            continue
            # else:
            # 判断面是否位于机身
            # for j in i.getVertices():
            # if (p.vertices[j].pointOn[0][2]>-2.87e3):
            # jishenFace=jishenFace+p.faces[i.index:i.index+1]
            # temp=1
            # break
            # 若面位于机身，则跳过骨架及蒙皮的判断
            # if temp==1:
            # continue
            outterFaces = outterFaces + p.faces[i.index:i.index + 1]
            outterNum = outterNum + 1
            p.Set(faces=p.faces[i.index:i.index + 1], name='outterStiff-%d' % outterNum)

    jishenFace = f.getByBoundingBox(-1501, -5001, -1, -999, 5001, 201)
    p.Set(faces=jishenFace, name='allJishen')
    p.Set(faces=tranStiff, name='allTranStiff')
    p.Set(faces=longStiff, name='allLongStiff')
    p.Set(faces=outterFaces, name='mengpi')


###############################################################################################################
defineSet()
# defineStringer()
###########################################################################################################
############################################################################
# session.viewports['Viewport: 1'].enableMultipleColors()
# session.viewports['Viewport: 1'].setColor(initialColor='#BDBDBD')
# cmap=session.viewports['Viewport: 1'].colorMappings['Set']
# session.viewports['Viewport: 1'].setColor(colorMapping=cmap)
# session.viewports['Viewport: 1'].disableMultipleColors()
#####################################################################################################################################################
'''
坐标: 顶点 3 :1.181818182E+03,3.E+03,200.
坐标: 顶点 10 :1.181818182E+03,1.5E+03,200.
坐标: 顶点 9 :2.818181818E+03,1.5E+03,200.
坐标: 顶点 33 :1.181818182E+03,0.,200.
坐标: 顶点 20 :2.818181818E+03,0.,200.
坐标: 顶点 24 :4.454545455E+03,0.,200.
坐标: 顶点 50 :4.5E+03,-1.5E+03,200.
坐标: 顶点 46 :2.818181818E+03,-1.5E+03,200.
坐标: 顶点 59 :1.181818182E+03,-1.5E+03,200.
坐标: 顶点 83 :1.181818182E+03,-3.E+03,200.
坐标: 顶点 73 :2.818181818E+03,-3.E+03,200.
坐标: 顶点 88 :1.181818182E+03,-3.909090909E+03,200.
'''

# MPC
a = mdb.models['Model-1'].rootAssembly
v1 = a.instances['Part-1-1'].vertices
loadPoint = v1.findAt(((1.181818182E+03, 3.E+03, 200.),), ((1.181818182E+03, 1.5E+03, 200.),),
                      ((2.818181818E+03, 1.5E+03, 200.),), ((1.181818182E+03, 0., 200.),),
                      ((2.818181818E+03, 0., 200.),), ((4.454545455E+03, 0., 200.),),
                      ((4.5E+03, -1.5E+03, 200.),), ((2.818181818E+03, -1.5E+03, 200.),),
                      ((1.181818182E+03, -1.5E+03, 200.),), ((1.181818182E+03, -3.E+03, 200.),),
                      ((2.818181818E+03, -3.E+03, 200.),), ((1.181818182E+03, -3.666666667E+03, 200.),), )
# 仅用于修改加载点后使用
vertic = p.vertices.getByBoundingBox(-999, -5000, 199, 5000, 5000, 201)
region2 = a.Set(vertices=loadPoint, name='loadPoint')

if loadCases == 1:
    refPoint = a.referencePoints.findAt((2000.0, 750.0, 100.0), )
elif loadCases == 2:
    refPoint = a.referencePoints.findAt((2000.0, -750.0, 100.0), )
elif loadCases == 3:
    refPoint = a.referencePoints.findAt((2000.0, -2250.0, 100.0), )

a.Set(referencePoints=(refPoint,), name='Set-refPoint')
# a.ReferencePoint(point=(2000.0, 750.0, 100.0))
# a.ReferencePoint(point=(2000.0, -750.0, 100.0))
# a.ReferencePoint(point=(2000.0, -2250.0, 100.0))


# 划分网格
elemType1 = mesh.ElemType(elemCode=S4, elemLibrary=STANDARD,
                          secondOrderAccuracy=OFF)
elemType2 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
faces = f[:]
pickedRegions = (faces,)
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
p.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)
p.generateMesh()

# >>> 修改点2：替换为改良版的 submitJob (防止删 weight.txt) <<<
name = 'loadcases%s-stiffNum' % (loadCases)
for i in num:
    name = name + '-' + str(i)


def submitJob(name):
    extensions = ['.odb', '.lck', '.sta', '.msg', '.log', '.dat', '.com', '.sim', '.prt', '.ipm']
    # 排除 weight.txt，只删仿真结果
    result_files = ['Displacement.txt', 'Mises-outterFaces.txt',
                    'Mises-originStiff.txt', 'Mises-allReinforcedStiff.txt']

    print('Checking for old files to clean...')
    for ext in extensions:
        filename = name + ext
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

    for txt_file in result_files:
        if os.path.exists(txt_file):
            try:
                os.remove(txt_file)
            except:
                pass

    # >>> 修改点：将 numCpus 和 numDomains 从 6 改为 4 (或者 2) <<<
    mdb.Job(name=name, model='Model-1', description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT,
            numCpus=4, numDomains=4, numGPUs=0)  # <--- 这里改成了 4

    print("Job submitted. Waiting for completion inside Abaqus...")
    mdb.jobs[name].submit(consistencyChecking=OFF)
    mdb.jobs[name].waitForCompletion()


def GetWeight():
    p = mdb.models['Model-1'].parts['Part-1']
    element = p.elements
    weight = p.getMassProperties(regions=element)['mass']
    outfile_name = open('weight.txt', 'w')
    outfile_name.write('%f \n' % weight)
    outfile_name.close()
    return weight


def maxMisesDis(name):
    stress = []
    maxValue = 0
    odb = openOdb(name + '.odb')
    step = odb.steps['Step-1']
    frame = step.frames[-1]
    allField = odb.steps['Step-1'].frames[-1].fieldOutputs

    # --- 修复开始：位移提取 ---
    # 位移
    stressSet = allField['U'].getSubset(region=odb.rootAssembly.instances['PART-1-1'])
    for stressValue in stressSet.values:
        # 使用 .magnitude 获取合位移（模长），这是 Abaqus 标准写法
        # 如果你原来的意图是取三个方向绝对值的最大值，应该用 max(map(abs, stressValue.data))
        # 但通常工程上都是看合位移 magnitude
        if stressValue.magnitude:
            stress.append(stressValue.magnitude)
        else:
            stress.append(0.0)

    maxValue = max(stress)
    outfile_name = open('Displacement.txt', 'w')
    outfile_name.write('%f \n' % maxValue)
    outfile_name.close()
    # --- 修复结束 ---

    # 蒙皮
    stress = []
    stressSet = allField['S'].getSubset(region=odb.rootAssembly.instances['PART-1-1'].elementSets['ALLOUTTERFACES'])
    for stressValue in stressSet.values:
        stress.append(stressValue.mises)

    maxValue = max(stress)
    outfile_name = open('Mises-outterFaces.txt', 'w')
    outfile_name.write('%f \n' % maxValue)
    outfile_name.close()
    # 初始加筋
    stress = []
    stressSet = allField['S'].getSubset(region=odb.rootAssembly.instances['PART-1-1'].elementSets['ALLORIGINSTIFF'])
    for stressValue in stressSet.values:
        stress.append(stressValue.mises)

    maxValue = max(stress)
    outfile_name = open('Mises-originStiff.txt', 'w')
    outfile_name.write('%f \n' % maxValue)
    outfile_name.close()
    # 后加墙
    if num.count(0) != 9:
        stress = []
        stressSet = allField['S'].getSubset(
            region=odb.rootAssembly.instances['PART-1-1'].elementSets['ALLREINFORCEDSTIFF'])
        for stressValue in stressSet.values:
            stress.append(stressValue.mises)

        maxValue = max(stress)
        outfile_name = open('Mises-allReinforcedStiff.txt', 'w')
        outfile_name.write('%f \n' % maxValue)
        outfile_name.close()
    odb.close()
    return


# 修改初始骨架位置
def reSketchOrigin(para_1, para_2, para_3, para_4, para_5, para_6, para_7):
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    # 横向
    s.Line(point1=(-3000, 3000 + para_1), point2=(5000, 3000 + para_1))
    s.Line(point1=(-3000, 1500 + para_2), point2=(5000, 1500 + para_2))
    s.Line(point1=(-3000, 0 + para_3), point2=(5000, 0 + para_3))
    s.Line(point1=(-3000, -3000 + para_4), point2=(5000, -3000 + para_4))
    s.Line(point1=(-3000, -1500 + para_5), point2=(5000, -1500 + para_5))
    # 纵向
    s.Line(point1=(-2500, 5000), point2=(-2500, -5000))
    s.Line(point1=(-300 + para_6, 5000), point2=(-300 + para_6, -5000))
    s.Line(point1=(-1300 + para_7, 5000), point2=(-1300 + para_7, -5000))
    mdb.models['Model-1'].ConstrainedSketch(name='Sketch-6', objectToCopy=s)
    mdb.models['Model-1'].sketches.changeKey(fromName='__profile__',
                                             toName='Sketch-origin')
    s.unsetPrimaryObject()
    return


# ==============================================================================
#                                  主执行逻辑
# ==============================================================================
name='XSJ-6'

# 1. 计算并保存重量
try:
   print("Calculating weight...")
   GetWeight()
except Exception as e:
   print("Error calculating weight: %s" % str(e))

# 2. 提交计算 (注意：submitJob 内部现在会通过 waitForCompletion 等待计算结束)
try:
   submitJob(name)
except Exception as e:
   print("Job submission failed: %s" % str(e))

# 3. 缓冲等待 (防止 ODB 文件锁死)
print("Analysis finished. Sleeping for 10 seconds to ensure file unlock...")
time.sleep(10)  # 这一步非常重要！不要删！

# 4. 提取结果
odb_file = name + '.odb'
if os.path.exists(odb_file):
   print("ODB file found. Extracting results...")
   try:
      maxMisesDis(name)
   except Exception as e:
      print("Post-processing failed: %s" % str(e))
else:
   print("Error: ODB file missing! Simulation might have failed.")

# 5. 保存模型
mdb.saveAs(pathName='XSJ-6_after_run.cae')