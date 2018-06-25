# -*- coding: cp1254 -*-
import arcpy,os
from arcpy.sa import *
import matplotlib.pyplot as plt
from arcpy import env
arcpy.env.overwriteOutput = True
arcpy.env.workspace = "c:/deneme"  

sus_map=arcpy.GetParameterAsText(0)
train=arcpy.GetParameterAsText(1)
test=arcpy.GetParameterAsText(2)

##sus_map="c:/deneme/sus_map.tif"
##train="c:/deneme/train.shp"
##test="c:/deneme/test.shp"

mx=float (arcpy.GetRasterProperties_management (sus_map, "MAXIMUM").getOutput (0))
mn=float (arcpy.GetRasterProperties_management (sus_map, "MINIMUM").getOutput (0))

e=(float(mx)-float(mn))/100

d=[]
x=0
y=0
z=0
for f in range (100):
    
    x=x+1
    y=mn+e
    z=z+mn
    q=[]
    q.append(z)
    q.append(y)
    q.append(x)
    d.append(q)
    mn=y
    z=0

total=Reclassify(sus_map,"VALUE",RemapRange(d),"NODATA")
total_exp="c:/deneme/total.tif"
total.save(total_exp)

trn=ExtractByMask(total,train)
train_exp="c:/deneme/train.tif"
trn.save(train_exp)

tes=ExtractByMask(total,test)
test_exp="c:/deneme/test.tif"
tes.save(test_exp)
##.............................................
arcpy.AddField_management(total_exp,"total","DOUBLE")
arcpy.AddField_management(total_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(total_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(total_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(total_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(total_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","total","NID"
with arcpy.da.SearchCursor(total_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        
arcpy.AddMessage(lst_count)
with arcpy.da.UpdateCursor(total_exp,"total") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##...........................................................................
arcpy.AddField_management(train_exp,"train","DOUBLE")
arcpy.AddField_management(train_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(train_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(train_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(train_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(train_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","train","NID"
with arcpy.da.SearchCursor(train_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        
arcpy.AddMessage(lst_count)

with arcpy.da.UpdateCursor(train_exp,"train") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##...........................................................
arcpy.AddField_management(test_exp,"test","DOUBLE")
arcpy.AddField_management(test_exp,"NID","LONG")

block="""rec=0
def yaz():
    global rec
    pstart=1
    pinterval=1
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression="yaz()"
arcpy.CalculateField_management(test_exp,"NID",expression,"PYTHON",block)
lst_nid=list()

with arcpy.da.SearchCursor(test_exp,"NID") as dd:
    for row in dd:
        lst_nid.append(row[0])
        
        del row
    
    del dd

mx=max(lst_nid)
crs=arcpy.da.InsertCursor(test_exp,["NID"])
for i in range(mx+1,101):
   crs.insertRow("0")
arcpy.CalculateField_management(test_exp,"NID",expression,"PYTHON",block)
lst_value=[]
lst_count=[]
lst_nc=[]
lst_nid_2=[]
sc_fields="value","count","test","NID"
with arcpy.da.SearchCursor(test_exp,sc_fields) as scur:
    for row in scur:
        lst_value.append(row[0])
        lst_count.append(row[1])
        lst_nc.append(row[2])
        lst_nid_2.append(row[3])                   
        del row

for i in range(len(lst_nid_2)):
    if lst_value[i]!=i+1:
        lst_value.insert(i,0)
    


h=0
for k in range (len(lst_nid_2)):
   
    if lst_value[k]!=lst_nid_2[k]:
        
        d=lst_count.insert(lst_nid_2[k]-1,0)
        
arcpy.AddMessage(lst_count)

with arcpy.da.UpdateCursor(test_exp,"test") as ucur:
            for row in ucur:
                row[0]=lst_count[h]
                ucur.updateRow(row)
                h=h+1
                del row
##..........................................................................


arcpy.JoinField_management(total_exp,"NID",train_exp,"NID","train")
arcpy.JoinField_management(total_exp,"NID",test_exp,"NID","test")
##........................................................................kümülatif toplam hesaplanıyor

arcpy.AddField_management(total_exp,"kum_total","DOUBLE")
arcpy.AddField_management(total_exp,"kum_train","DOUBLE")
arcpy.AddField_management(total_exp,"kum_test","DOUBLE")

block2="""rec=0
def kum_tot(r):
    global rec
    pstart=r
    pinterval=r
    if(rec==0):
        rec=pstart
    else:
        rec+=pinterval
    return rec"""
expression2="kum_tot(!total!)"
arcpy.CalculateField_management(total_exp,"kum_total",expression2,"PYTHON",block2)
arcpy.CalculateField_management(total_exp,"kum_train","kum_tot(!train!)","PYTHON",block2)
arcpy.CalculateField_management(total_exp,"kum_test","kum_tot(!test!)","PYTHON",block2)
tot_fields="kum_total","kum_train","kum_test"
lst_tot=[]
lst_tr=[]
lst_tst=[]
with arcpy.da.SearchCursor(total_exp,tot_fields) as scur2:
    for row in scur2:
        lst_tot.append(row[0])
        lst_tr.append(row[1])
        lst_tst.append(row[2])
        del row
    del scur2

toplam_tot=max(lst_tot)
toplam_tr=max(lst_tr)
toplam_tst=max(lst_tst)
arcpy.AddMessage(toplam_tot)
arcpy.AddMessage(toplam_tr)
arcpy.AddMessage(toplam_tst)
##......................................................................
arcpy.AddField_management(total_exp,"c_tot","DOUBLE")
arcpy.AddField_management(total_exp,"c_tr","DOUBLE")
arcpy.AddField_management(total_exp,"c_tst","DOUBLE")
c="kum_total","kum_train","kum_test","c_tot","c_tr","c_tst"
with arcpy.da.UpdateCursor(total_exp,c) as ucur2:
    for row in ucur2:
        v=row[0]/toplam_tot
        k=row[1]/toplam_tr
        l=row[2]/toplam_tst
        row[3]=1-v
        row[4]=1-k
        row[5]=1-l
        ucur2.updateRow(row)
y="c_tot","c_tr","c_tst"
tot=[]
tr=[]
ts=[]
with arcpy.da.SearchCursor(total_exp,y) as scur2:
    for row in scur2:
        tot.append(row[0])
        tr.append(row[1])
        ts.append(row[2])
        del row
    del scur2

tot.insert(0,1)
tr.insert(0,1)
ts.insert(0,1)

tr_son=[]
ts_son=[]
for i in range(100):
    b=tot[i]-tot[i+1]
    n=tr[i]
    m=ts[i]
    p=b*n
    t=b*m
    tr_son.append(p)
    ts_son.append(t)
f=round(sum(tr_son)*100,2)
g=round(sum(ts_son)*100,2)
arcpy.AddMessage(sum(tr_son)*100)
arcpy.AddMessage(sum(ts_son)*100)
sc=plt.plot(tot,tr,color="red",label=":Success Rate"+"("+str(f)+")")
#plt.legend()
pr=plt.plot(tot,ts,color="blue",label=":Prediction Rate"+"("+str(g)+")")

plt.xlabel("1-Specifity")
plt.ylabel("Sensitivity")
plt.legend(loc="lower right")

plt.savefig("c:/deneme/polat.png",dpi=150)        
##kümülatif toplam
##rec=0
##def ali(r):
##   
##    global rec
##    pstart=r
##    pinterval=r
##    if(rec==0):
##        rec=pstart
##    else:
##        rec+=pinterval
##    return rec

arcpy.AddMessage("İŞLEM TAMAM")
