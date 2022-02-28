import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from ttkwidgets import CheckboxTreeview
import os

from sklearn.cluster import KMeans
from kneed import KneeLocator
import prince

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE
import seaborn as sns

from sklearn import metrics
from scipy.spatial.distance import cdist


def only_numbers(char):
    return char.isdigit()


def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select A File",
                                          filetype=(("xlsx files", "*.xlsx"),("All Files", "*.*")))
    label_file["text"] = filename
    # label_file["text"] = "C:\\Users\\Fdesk\\Desktop\\WORKSPACE\\WebMining\\Linkedin_4k_200.xlsx"
    return None


def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    global df
    clear_data()
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename)
        else:
            df = pd.read_excel(excel_filename,index_col=[0])

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) # let the column heading = column name

    df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_rows:
        tv1.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
        
    df_desc = df.describe(include = ["object"])
    df_desc.insert(0, '', df.describe(include = ["object"]).index)
    
    
    tv2["column"] = list(df_desc.columns)
    tv2["show"] = "headings"
    for column in tv2["columns"]:
        tv2.heading(column, text=column) # let the column heading = column name

    df_desc_rows = df_desc.to_numpy().tolist() # turns the dataframe into a list of lists
    for row in df_desc_rows:
        tv2.insert("", "end", values=row) # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
        
    
    if df is not None:
        
    
        preSelectedCols = ['COMPANY','EDUCATIONS_0', 'FIELD_0',
       'DEGREE/COURSE_0', 'LICENSE/CERTS_0', 'ISSUING_AUTHORITY_0']
    
        for col in df.columns: 
            if col in preSelectedCols:
                ctX.insert('', 'end', iid = col, text=col, tags=["checked"])
            else:
                ctX.insert('', 'end', iid = col, text=col, tags=["unchecked"])
            
        for col in df.columns: 
            if col == "POSITION":
                ctY.insert('', 'end', iid = col, text=col, tags=["checked"])
            else:
                ctY.insert('', 'end', iid = col, text=col, tags=["unchecked"])
        
    return None


def plotEigensAndReturnBest(X_raw):
  try:
      mca = prince.MCA(n_components = 40, random_state=42)
      mca.fit_transform(X_raw)
      y = mca.explained_inertia_
      # y = np.cumsum(mca.explained_inertia_)
    
      xi = np.arange(1, len(y)+1, step=1)
      kneedle = KneeLocator(np.arange(1, len(y)+1, step=1), y, curve="concave", direction="decreasing")
      # print(kneedle.knee)
    
      plt.rcParams["figure.figsize"] = (10,6)
      fig, ax = plt.subplots()
    
      # y = np.cumsum(pca.explained_variance_ratio_)
    
      plt.plot(xi, y, marker='o', linestyle='--', color='r')
      plt.xlabel('Number of Components')
      plt.xticks(xi) #change from 0-based array index to 1-based human-readable label
      plt.ylabel('Eigen Values')
      plt.title('Eigen values of components')
      # plt.scatter(x= kneedle.knee, y= mca.eigenvalues_[kneedle.knee])
    
      plt.scatter(x=kneedle.knee, y=y[kneedle.knee-1] , s=300,  color='blue',marker="o",label="Elbow/Knee Point")
      ax.grid(axis='both')
      filename = "mca_eigens.png"
      plt.savefig(filename)
      entryMCA.set(kneedle.knee)
      tk.messagebox.showinfo(title="info", message = "MCA component file \'"+filename+"\' is saved. Optimal number of components is chosen as "+ str(kneedle.knee))
      return kneedle.knee 
  except Exception as e:
      tk.messagebox.showerror(title="info", message = str(e))
      print(e)
      clear_data()
      return None
  



def ApplyMCA():
    if df is None:
        print("NOOOONE")
    selectedCols = ctX.get_checked()
    yCol = ctY.get_checked()[0]
    X_raw = df[selectedCols]
    # print(selectedCols)
    best = plotEigensAndReturnBest(X_raw)

def extractMCAComponents():
    selectedCols = ctX.get_checked()
    X_raw = df[selectedCols]
    components = int(entryMCA.get())
    
    mca = prince.MCA(n_components = components,random_state=42)
    mca.fit(X_raw)
    global X
    global Y
    X = mca.transform(X_raw)
    Y = df[ctY.get_checked()[0]].values
    
    tk.messagebox.showinfo(title="info", message = "Feature Extraction is completed.")
    
    return None    
  
def plotElbowandReturnBest(X,clusterRange = np.arange(5, 125, 5)):
  
    wcss = {}
    clusterRange = np.arange(5, 125, 5)
    for k in clusterRange:
        kmeans = KMeans(n_clusters=k,random_state=42).fit(X)
        wcss[k]= sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'),axis=1)) / X.shape[0]
  
    xi = list(wcss.keys())
    y = list(wcss.values())
  
    elbow = KneeLocator(xi, y,  curve="convex", direction="decreasing")
    # print(elbow.elbow)
  
    plt.rcParams["figure.figsize"] = (8,6)
    fig, ax = plt.subplots()
    plt.plot(xi, y, marker='o', linestyle='--', color='r')
    plt.xlabel('k')
    plt.xticks(clusterRange) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('sum of square error')
    plt.title('WCSS')
  
    plt.scatter(x=elbow.elbow, y=wcss[elbow.elbow], s=300,  color='blue',marker="o",label="Elbow/Knee Point")
    ax.grid(axis='both')
    
    filename = "kmeans_elbow.png"
    plt.savefig(filename)
    entryKMeans.set(elbow.elbow)
    tk.messagebox.showinfo(title="info", message = "Kmeans elbow file \'"+filename+"\' is saved. The number of \'k\' is chosen as "+ str(elbow.elbow))
    
    return elbow.elbow
  # except Exception as e:
  #     tk.messagebox.showerror(title="info", message = str(e))
  #     print(e)
  #     clear_data()
  #     return None
  
def applyElbow():
    global k
    if X is None:
        tk.messagebox.showerror(title="error", message = "Feature extraction must be done first.")
        return None
    else:
        k = plotElbowandReturnBest(X)
    return None    
  
def applyKmeans():
    try:
        bestK = int(entryKMeans.get())
        kmeans = KMeans(n_clusters=bestK,random_state=42).fit(X)
        # print(np.unique(kmeans.labels_, return_counts=True))
        sil_score = metrics.silhouette_score(X, kmeans.labels_)
        calinski_score = metrics.calinski_harabasz_score(X, kmeans.labels_)
        hom_score = metrics.homogeneity_score(Y, kmeans.labels_)
        comp_score = metrics.completeness_score(Y, kmeans.labels_)
        # print(sil_score, hom_score, comp_score)
        df["KM"] = kmeans.labels_
        label_KM_res["text"] = "Silhouette Score = %0.4f\nCalinski Harabasz Score = %0.4f \nHomogeneity Score = %0.4f \nCompleteness Score = %0.4f " % (sil_score, calinski_score,hom_score, comp_score)
    except Exception as e:
        tk.messagebox.showerror(title="error", message = str(e))
    return None

def applyGMM():
    try:
        comps = int(entryGMM.get())
        print(comps)
        gm = GaussianMixture(n_components=30, random_state=0).fit(X)
        gm_labels_  = gm.predict(X)
        # print(np.unique(gm_labels_, return_counts=True))
        sil_score = metrics.silhouette_score(X, gm_labels_)
        calinski_score = metrics.calinski_harabasz_score(X, gm_labels_)
        hom_score = metrics.homogeneity_score(Y, gm_labels_)
        comp_score = metrics.completeness_score(Y, gm_labels_)
        df["GM"] = gm_labels_
        
        label_GMM_res["text"] = "Silhouette Score = %0.4f\nCalinski Harabasz Score = %0.4f \nHomogeneity Score = %0.4f \nCompleteness Score = %0.4f " % (sil_score, calinski_score,hom_score, comp_score)
    except:
        tk.messagebox.showwarning(title="warning",message="Değer giriniz.")
    return None

def applyDBSCAN():
    try:
        eps = float(entryEPS.get())
        minsample = int(entryMinSample.get())
        dbscan = DBSCAN(eps=eps, min_samples=minsample).fit(X)
        sil_score = metrics.silhouette_score(X, dbscan.labels_)
        calinski_score = metrics.calinski_harabasz_score(X, dbscan.labels_)
        hom_score = metrics.homogeneity_score(Y, dbscan.labels_)
        comp_score = metrics.completeness_score(Y, dbscan.labels_)
        df["DB"] = dbscan.labels_
        label_DBSCAN_res["text"] = "Silhouette Score = %0.4f\nCalinski Harabasz Score = %0.4f \nHomogeneity Score = %0.4f \nCompleteness Score = %0.4f " % (sil_score,calinski_score, hom_score, comp_score)
        
    except Exception as e:
        tk.messagebox.showerror(title="error", message = str(e))
    return None

def clear_data():
    df=None
    tv1.delete(*tv1.get_children())
    tv2.delete(*tv2.get_children())
    ctX.delete(*ctX.get_children())
    ctY.delete(*ctY.get_children())
    # ct.delete(*ct.get_children())
    return None



def run_tSNE():
    np.random.seed(42)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_res = tsne.fit_transform(X)
    
    df["tsne-one"] = tsne_res[:,0]
    df["tsne-two"] = tsne_res[:,1]

def plot_tSNE(targetField):
    figure = Figure(figsize=(5,5))
    ax = figure.subplots()
    ax.clear()
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue=targetField,
        palette=sns.color_palette("hls", n_colors = len(df[targetField].value_counts())),
        data=df,
        s=50,
        legend=None,
        ax = ax
        ).set_title("2D tSNE manifold for "+targetField)
    return figure

# def _clear(canvas):
#     for item in canvas.get_tk_widget().find_all():
#        canvas.get_tk_widget().delete(item)

def clearFrame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def plot_All():
    try:
        
        if df["KM"] is not None:
            clearFrame(frame_vis_KM)
            fig = plot_tSNE("KM")
            canvas = FigureCanvasTkAgg(fig, master= frame_vis_KM)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except Exception as e:
             tk.messagebox.showerror(title="error", message = str(e))
             print(e)
             pass   
    try:
        if df["DB"] is not None:
            clearFrame(frame_vis_DBSCAN)
            fig = plot_tSNE("DB")
            canvas = FigureCanvasTkAgg(fig, master= frame_vis_DBSCAN)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except Exception as e:
        tk.messagebox.showerror(title="error", message = str(e))
        print(e)
        pass
    try:
        if df["GM"] is not None:
            clearFrame(frame_vis_GMM)
            fig = plot_tSNE("GM")
            canvas = FigureCanvasTkAgg(fig, master= frame_vis_GMM)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except Exception as e:
        tk.messagebox.showerror(title="error", message = str(e))
        print(e)
        pass
    try:
        if ctY.get_checked()[0] is not None:
            clearFrame(frame_vis_GT)
            fig = plot_tSNE("POSITION")
            canvas = FigureCanvasTkAgg(fig, master= frame_vis_GT)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    except Exception as e:
        tk.messagebox.showerror(title="error", message = str(e))
        print(e)
        pass


    return None

# initalise the tkinter GUI
root = tk.Tk()

w = 1400
h = 1000

root.geometry(str(w)+"x"+str(h)) # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.

root.resizable(0, 0) # makes the root window fixed in size.

# v = Scrollbar(root)
# v.pack(side = RIGHT, fill=Y )


frame_file = tk.LabelFrame(root, text="Open File")
frame_file.place(height=60, width=w)

frame_data = tk.LabelFrame(root, text="Excel Data")
frame_data.place(height=160, width=w, y= 61, x = 0)

frame_describe = tk.LabelFrame(root, text="Data Summary")
frame_describe.place(height=140, width=w, y= 221, x = 0)

frame_check = tk.LabelFrame(root, text="Data columns(X,Y)")
frame_check.place(height=150, width=w, y =361, x=0)

frame_featext = tk.LabelFrame(root, text="Feature Extraction(Multiple Correspondence Analysis - MCA)")
frame_featext.place(height=50, width=w, y =511, x=0)

frame_clustering = tk.LabelFrame(root, text="Clustering")
frame_clustering.place(height=150, width=w, y =561, x=0)

frame_KM = tk.LabelFrame(frame_clustering, text="K-Means")
frame_KM.place(relheight=1, relwidth=0.333, rely= 0, relx=0)

frame_GMM = tk.LabelFrame(frame_clustering, text="Gaussian Mixture Model")
frame_GMM.place(relheight=1, relwidth=0.333, rely= 0, relx=0.333)

frame_DBSCAN = tk.LabelFrame(frame_clustering, text="DBSCAN")
frame_DBSCAN.place(relheight=1, relwidth=0.333, rely= 0, relx=0.666)

frame_visualization = tk.LabelFrame(root, text="Visualization(T-SNE 2D)")
frame_visualization.place(height=290, width=w, x=0, y= 711)

frame_vis_KM = tk.LabelFrame(frame_visualization, text="K-Means")
frame_vis_KM .place(relheight=1, relwidth=0.25, rely= 0, relx=0)

frame_vis_GMM = tk.LabelFrame(frame_visualization, text="Gaussian Mixture Model")
frame_vis_GMM.place(relheight=1, relwidth=0.25, rely= 0, relx=0.25)

frame_vis_DBSCAN = tk.LabelFrame(frame_visualization, text="DBSCAN")
frame_vis_DBSCAN.place(relheight=1, relwidth=0.25, rely= 0, relx=0.50)

frame_vis_GT = tk.LabelFrame(frame_visualization, text="True Label")
frame_vis_GT.place(relheight=1, relwidth=0.25, rely= 0, relx=0.75)


# Buttons
button1 = tk.Button(frame_file, text="Browse A File", command=lambda: File_dialog())
button1.place(rely=0.3, relx=0.50)

button2 = tk.Button(frame_file, text="Load File", command=lambda: Load_excel_data())
button2.place(rely=0.3, relx=0.30)

# The file/file path text
label_file = ttk.Label(frame_file, text="No File Selected")
label_file.place(rely=0, relx=0)


## Treeview Widget
tv1 = ttk.Treeview(frame_data)
tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame_data).

treescrolly = tk.Scrollbar(frame_data, orient="vertical", command=tv1.yview) # command means update the yaxis view of the widget
treescrollx = tk.Scrollbar(frame_data, orient="horizontal", command=tv1.xview) # command means update the xaxis view of the widget
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget
treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

tv2 = ttk.Treeview(frame_describe)
tv2.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame_describe).

treescrolly2 = tk.Scrollbar(frame_describe, orient="vertical", command=tv2.yview) # command means update the yaxis view of the widget
treescrollx2 = tk.Scrollbar(frame_describe, orient="horizontal", command=tv2.xview) # command means update the xaxis view of the widget
tv2.configure(xscrollcommand=treescrollx2.set, yscrollcommand=treescrolly2.set) # assign the scrollbars to the Treeview Widget
treescrollx2.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
treescrolly2.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget


ctX = CheckboxTreeview(frame_check, show='tree') # hide tree headings
# ctX.grid(row=0,column=0)
ctX.place(relx=0, rely=0, relheight=1, relwidth=0.5) # set the height and width of the widget to 100% of its container (frame_data).

ctY = CheckboxTreeview(frame_check, show='tree') # hide tree headings
# ctY.grid(row=0,column=1)
# ctY.state(["disabled"])
ctY.place(relx=0.5, rely=0, relheight=1, relwidth=0.5) # set the height and width of the widget to 100% of its container (frame_data).

button3 = tk.Button(frame_check, text="Select columns and Apply MCA", command= lambda: ApplyMCA())
button3.place(relx=0.35, rely=0.7)

label_mca = ttk.Label(frame_featext, text="Number of components")
label_mca.place(y=0, x=0)

validation = root.register(only_numbers)

entryMCA = tk.StringVar()
entry_mca = tk.Entry(frame_featext,textvariable=entryMCA, validate="key", validatecommand=(validation, '%S'))
entry_mca.place(width = 150,x=150,y=0)

button4 = tk.Button(frame_featext, text="Extract Feautres", command=lambda: extractMCAComponents())
button4.place(y=0, x=311, height=20)

# label_mca_info = ttk.Label(frame_featext, text="-")
# label_mca_info.place(y=0, x=400)


label_km = ttk.Label(frame_KM, text="Number of clusters(k)")
label_km.place(y=0, x=0)

entryKMeans = tk.StringVar()
entry_KM = tk.Entry(frame_KM,textvariable=entryKMeans, validate="key", validatecommand=(validation, '%S'))
entry_KM.place(width = 150,x=150,y=0)
button5 = tk.Button(frame_KM, text="Apply", command=lambda: applyKmeans())
button5.place(y=0, x=311, height=20)
button6 = tk.Button(frame_KM, text="Use Elbow", command=lambda: applyElbow())
button6.place(y=0, x=380, height=20)

label_KM_res = ttk.Label(frame_KM, text="")
label_KM_res.place(y=45, x=0)

label_GMM_res = ttk.Label(frame_GMM, text="")
label_GMM_res .place(y=45, x=0)

label_DBSCAN_res = ttk.Label(frame_DBSCAN, text="")
label_DBSCAN_res.place(y=45, x=0)


label_noc = ttk.Label(frame_GMM, text="Number of Components(k)")
label_noc.place(y=0, x=0)
entryGMM = tk.StringVar()
entry_GMM = tk.Entry(frame_GMM, textvariable=entryGMM, validate="key", validatecommand=(validation, '%S'))
entry_GMM.place(width = 150,x=150,y=0)
button7 = tk.Button(frame_GMM, text="Apply", command=lambda: applyGMM())
button7.place(y=0, x=311, height=20)


label_eps = ttk.Label(frame_DBSCAN, text="EPS")
label_eps.place(y=0, x=0)

entryEPS = tk.StringVar()
entry_EPS = tk.Entry(frame_DBSCAN, textvariable=entryEPS)
entry_EPS.place(width = 150,x=150,y=0)

label_mins = ttk.Label(frame_DBSCAN, text="Min Samples")
label_mins.place(y=25, x=0)

entryMinSample = tk.StringVar()
entry_MinSample = tk.Entry(frame_DBSCAN, textvariable=entryMinSample, validate="key", validatecommand=(validation, '%S'))
entry_MinSample.place(width=150,x=150,y=25)
button8 = tk.Button(frame_DBSCAN, text="Apply", command=lambda: applyDBSCAN())
button8.place(y=0, x=311, height=20)

button9 = tk.Button(frame_visualization, text="Run tSNE", command=lambda: run_tSNE())
button9.place(y=-20, x=140)

button10 = tk.Button(frame_visualization, text="plot", command=lambda: plot_All())
button10.place(y=-20, x=220)

root.mainloop()
