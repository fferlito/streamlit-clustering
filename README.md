# Streamlit clustering app

Link to app: https://fferlito-streamlit-clustering-app-s41ykh.streamlit.app/

This web app allows the user to upload a raster file, make quick changes to it (i.e. brightness), run K-means and download the resulting raster file.
The output label will be stored in the first band of the output raster.

Before uploading a raster, it's reccomended to scale the input raster (i.e. with GDAL translate) and to convert the numbers to uint8 (bytes), as streamlit has very low memory limits
