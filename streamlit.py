import sys, os
import numpy as np
import streamlit as st
from digitalcore import GeochemClusterExperiment


# Caches so that it is not rerun every time the app is rerun
@st.experimental_memo
def create_experiment(uf):
    return GeochemClusterExperiment.read_csv(uf)


def make_element_image(element, experiment):
    fig = experiment.plot_element(element, labels=True)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def main():

    st.title("Geochemistry Cluster Explorer")

    uf = st.file_uploader("Upload your data file", type="csv")

    if uf is not None:

        experiment = create_experiment(uf)
        st.dataframe(experiment.data)

        features = experiment.get_features()
        st.text(f'Detected {len(features)} features:')

        st.dataframe({'feature': sorted(features)})

        experiment.prepare(silent=True)

        n_clusters = st.select_slider("Choose number of clusters", 
                                     options=range(2, 10), value=4)


        if st.button("Submit"):

            experiment.add("hclust", {'num_clusters': n_clusters})
            experiment.create()
            experiment.label()

            experiment.plottype = "elbow"
            experiment.plotmodel(display_format='streamlit')

            
            element = st.selectbox(
                "Choose an element to plot", 
                sorted(set([f.split('_')[0] for f in features])),
            )

            def plot_element():
                image = make_element_image(element, experiment)
                st.image(image, caption='Sunrise by the mountains')

            st.button("Plot", on_click=plot_element())




if __name__ == "__main__":
    main()