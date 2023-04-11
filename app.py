import streamlit as st
from streamlit_juxtapose import juxtapose

from PIL import Image, ImageEnhance
import requests
from tempfile import NamedTemporaryFile
from sklearn import cluster
from scipy import ndimage
import numpy
import pathlib
import matplotlib.pyplot as plt
import PIL
import uuid
import rioxarray as xr

STREAMLIT_STATIC_PATH = (
    pathlib.Path(st.__path__[0]) / "static"
)  # at venv/lib/python3.9/site-packages/streamlit/static


uploaded = st.file_uploader("Upload a file", type=["tif"])

with st.sidebar:
    st.subheader("Quick changes")
    brightness = st.slider('Brightness', 0.0, 5.0, 1.0, 0.01)
    constrast = st.slider('Contrast', 0.0, 3.0, 1.0, 0.01)

IMG1 = str(uuid.uuid1()) + ".png"
IMG2 = str(uuid.uuid1()) + ".png"

if uploaded:
    with NamedTemporaryFile("wb", suffix=".tif") as f:
        f.write(uploaded.getvalue())

        # f.name is the path of the temporary file
        im = Image.open(f.name).convert('RGB')

        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(constrast)


        r, g, b = im.split()
        r = r.point(lambda i: i * (brightness))
        g = g.point(lambda i: i * (brightness))
        b = b.point(lambda i: i * (brightness))
        im = Image.merge('RGB', (r, g, b))
        im.save(STREAMLIT_STATIC_PATH / IMG1)
        ds = xr.open_rasterio(f.name)
        ds.rio.to_raster('input.tif')

        #st.image(im, caption=' ', channels='RGB')
        generate_output = False
        with st.sidebar:
            with st.form("my_form"):
                st.subheader("Run clustering")
                n_k = st.slider('Indicate number of clusters', 2, 20, 4, 1)
                submitted = st.form_submit_button("Submit")
                if submitted:

                    # Split into 3 channels
                    r, g, b = im.split()

                    # Increase Reds
                    r = r.point(lambda i: i * (brightness))
                    g = g.point(lambda i: i * (brightness))
                    b = b.point(lambda i: i * (brightness))

                    # Recombine back to RGB image
                    im = Image.merge('RGB', (r, g, b))
                    im = numpy.asarray(im)
                    shape_x, shape_y, shape_z = im.shape
                    image_2d = im.reshape(shape_x*shape_y, shape_z)

                    kmeans_cluster = cluster.KMeans(n_clusters=n_k)
                    kmeans_cluster.fit(image_2d)
                    cluster_centers = kmeans_cluster.cluster_centers_
                    cluster_labels = kmeans_cluster.labels_

                    cm = plt.get_cmap('gist_rainbow')
                    NUM_COLORS = n_k
                    color= numpy.array([list(cm(1.*i/NUM_COLORS)) for i in range(NUM_COLORS)]) * 255
                    reshaped = numpy.uint8(color[cluster_labels].reshape(shape_x, shape_y, 4))
                    im = PIL.Image.fromarray(reshaped).convert('RGB')
                    im.save(STREAMLIT_STATIC_PATH / IMG2)
                    ds[dict(band=0)] = cluster_labels.reshape(shape_x, shape_y)
                    ds[dict(band=1)] = reshaped[:,:,1]
                    ds[dict(band=2)] = reshaped[:,:,2]
                    ds.rio.to_raster('output.tif')
                    generate_output = True
                else:
                    generate_output = False



        if generate_output:
            juxtapose(IMG1, IMG2, 1500)
        else:
            st.image(im)

                #st.image(im, caption='Clusters')
