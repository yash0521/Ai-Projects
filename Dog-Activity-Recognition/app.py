import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from helpers_train import load_train_dataset
from helpers_train import load_train_dataset
from helpers_test import process_raw_csv_data, test_data_preprocessing, make_prediction, reshape_features, ensemble_voting


def main():
    st.sidebar.title("Dog Activity Recognizer")

    main_menu = st.sidebar.radio(
        "Select an option",
        ("App Insights", "Quick Guide", "Application")
    )
    if main_menu == "App Insights":
        st.title("Paper")

        paper_drive_link = "https://drive.google.com/file/d/1rOQl03EHiVCTFlE10RVPsoQnZODZbScr/view?usp=drive_link"

        text = f"Click <a href='{paper_drive_link}' target='_blank'>here</a> to view the paper."
        st.markdown(
            f"<p style='font-size: 26px;'>{text}</p>", unsafe_allow_html=True)

        st.title("Github Repo")

        git_repo_link = "https://github.com/ProjectFantastic3/Dog-Activity-Recognition"

        text = f"Click <a href='{git_repo_link}' target='_blank'>here</a> to view the code."
        st.markdown(
            f"<p style='font-size: 26px;'>{text}</p>", unsafe_allow_html=True)

        st.title("Data Analysis")

        st.markdown(
            "<h1 style='text-align: center; color: #008080; font-size: 26px;'>Data Information</h1>", unsafe_allow_html=True)
        column_info = {
            'ax': 'Acceleration X: Represents acceleration along the X-axis. Positive values indicate acceleration in the positive X-direction, while negative values indicate acceleration in the negative X-direction.',
            'ay': 'Acceleration Y: Indicates acceleration along the Y-axis. Positive values signify acceleration in the positive Y-direction, and negative values denote acceleration in the negative Y-direction.',
            'az': 'Acceleration Z: Describes acceleration along the Z-axis. Positive values denote acceleration in the positive Z-direction, and negative values indicate acceleration in the negative Z-direction.',
            'wx': 'Angular Velocity X: Represents angular velocity around the X-axis, indicating the rate of rotation around the X-axis.',
            'wy': 'Angular Velocity Y: Signifies angular velocity around the Y-axis, indicating the rate of rotation around the Y-axis.',
            'wz': 'Angular Velocity Z: Represents angular velocity around the Z-axis, indicating the rate of rotation around the Z-axis.',
            'AngleX': 'Angle X: Denotes orientation angle with respect to the X-axis, providing information about tilt or inclination along the X-axis.',
            'AngleY': 'Angle Y: Indicates orientation angle with respect to the Y-axis, providing information about tilt or inclination along the Y-axis.',
            'AngleZ': 'Angle Z: Represents orientation angle with respect to the Z-axis, providing information about rotation or tilt along the Z-axis.'
        }
        for column, info in column_info.items():
            st.subheader(column)
            st.write(info)
            st.write('\n')

        dataframe = {
            'ax': [-1.2190, -0.8440, 0.0580, 0.8145, -0.1509, -2.2520, -0.0933, -0.7860, -0.7340, 0.1392],
            'ay': [-0.1490, 0.1020, 0.1830, 0.7466, -0.7368, 0.1710, 0.8467, 0.1250, 0.0650, -0.8584],
            'az': [-2.5200, -0.5250, -0.9870, 0.7510, -0.6685, -2.0440, 0.4863, -0.6090, -0.7080, -0.2539],
            'wx': [-0.6710, -1.5870, -0.9160, -73.0591, 25.7568, 0.6710, -0.1221, -2.6250, -0.7930, -114.8071],
            'wy': [116.0280, -7.5070, 0.4270, 45.7153, -21.4844, 7.9960, 0.5493, -7.9960, -1.0990, -11.1694],
            'wz': [71.1670, -0.3050, 0.3050, 108.4595, -35.1563, -29.7850, 2.8687, 2.7470, -3.4180, 126.8921],
            'angleX': [167.1510, 168.7660, 168.8320, 65.2203, -109.9567, -167.4370, 59.6503, 165.3440, 173.1880, -137.5598],
            'angleY': [16.2430, 57.1730, -3.6860, -30.9540, 3.5925, 43.7530, 5.4108, 52.3830, 45.1210, 17.4078],
            'angleZ': [-130.0120, 60.1890, -20.5660, -89.7473, 138.5706, 82.5730, 8.2452, -84.7430, 20.3850, 32.6294],
            'label': ['running', 'sitting', 'lying', 'walking', 'climbing', 'walking', 'sitting', 'sitting', 'sitting', 'climbing']
        }

        # folder_path = "processed_data/train"
        # data_df = load_train_dataset(folder_path)
        data_df = pd.DataFrame(dataframe)

        # Print out the first 10 rows of the data
        # Give a title to the table with big font size
        st.write("Sample Data")
        st.write(data_df.sample(10))

        st.write("Training Data Shape")
        st.write((285905, 10))

        # st.write("Label Distribution")
        label_Distribution = plt.imread("utils/Images/label_Distributions.png")
        st.image(label_Distribution, caption="")

        # Line chart of all actions
        st.write("➣ Line chart of training data for each actions over time")

        lineplot_walking = plt.imread("utils/Images/LinePlot_walking.png")
        st.image(lineplot_walking, caption="")

        lineplot_running = plt.imread("utils/Images/LinePlot_running.png")
        st.image(lineplot_running, caption="")

        lineplot_sitting = plt.imread("utils/Images/LinePlot_sitting.png")
        st.image(lineplot_sitting, caption="")

        lineplot_lying = plt.imread("utils/Images/LinePlot_lying.png")
        st.image(lineplot_lying, caption="")

        lineplot_climbing = plt.imread("utils/Images/LinePlot_climbing.png")
        st.image(lineplot_climbing, caption="")

        st.write("➣ Scatter plot of all training data for all actions")
        scatter_plot = plt.imread("utils/Images/scatter_plot.png")
        st.image(scatter_plot, caption="")

        # pairwise_plot = plt.imread("utils/Images/Pairwise_scatter.png")
        # st.image(pairwise_plot, caption="")

        st.write("➣ Correlation matrix of all training data for all actions")
        correlation_matrix = plt.imread("utils/Images/Correlation_matrix.png")
        st.image(correlation_matrix, caption="")

        st.title("Evaluations")

        st.write("➣ Classification Report and Confusion Matrix of CNN Model")
        classification_report = plt.imread(
            "utils/Images/classification_report.png")
        st.image(classification_report, caption="",
                 width=800, use_column_width=False)

        # Confusion Matrix
        confusion_matrix = plt.imread("utils/Images/confusion_matrix.png")
        st.image(confusion_matrix, caption="",
                 width=800, use_column_width=False)

    elif main_menu == "Quick Guide":
        flow_chart = plt.imread("utils/Images/Flowchart.png")
        st.image(flow_chart, caption="")

        # st.title("See How It Works.. ")

        # Video
        # video_file = open('utils/video.mp4', 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

        # Step 1
        st.title("Step 1: Attach the device to your dog's collar")

        # Image
        image = Image.open("utils/Images/guide_step1.jpg")
        image = image.resize((400, 400))
        st.image(image, caption="", use_column_width=False)

        st.write("➣ Equip the dog with a motion sensor to capture its activity data.")
        st.write("➣ Attach the motion sensor securely to the dog's collar.")
        st.write(
            "➣ Make sure the motion sensor is turned on and the battery is charged.")

        # Step 2
        st.title("Step 2: Record the dog's activity data using the motion sensor")

        # Image
        image = Image.open("utils/Images/guide_step2.png")
        image = image.resize((800, 400))
        st.image(image, caption="", use_column_width=False)

        st.write(
            "➣ Establish a bluetooth connection between the motion sensor and wit motion app. ")
        st.write(
            "➣ Select the device and port number from the list of available devices and ports.")
        st.write(
            "➣ Record the dog's activity data using the motion sensor by pressing the record button.")
        st.write(
            "➣ Stop recording when you want to stop and save the data in a csv file.")

        # Step 3
        st.title("Step 3: Upload the csv file to the Dog Activity Recognizer App")

        # Image
        image = Image.open("utils/Images/guide_step3.png")
        image = image.resize((800, 400))
        st.image(image, caption="", use_column_width=False)
        smaple_csv_glink = "https://drive.google.com/file/d/1HB3OHQ-vVlR-AA2erIBRCukgI28BhDP2/view?usp=drive_link"

        st.write("➣ Upload the csv file to the Dog Activity Recognizer App.")
        st.write("➣ Make sure the csv file is in the correct format.")
        st.write("➣ The app will analyze the data and predict the dog's activity.")
        st.write(
            f"➣ If you don't have a csv file, you can download a sample csv file from [here]({smaple_csv_glink}).")

        # Step 4
        st.title("Step 4: Data Processing and Prediction")

        # Image
        image = Image.open("utils/Images/guide_step4.png")
        image = image.resize((800, 400))
        st.image(image, caption="", use_column_width=False)

        st.write("➣ The app will process the data and make predictions.")
        st.write("➣ The app will display the prediction results.")

        # Step 5
        st.title("Step 5: View the prediction results")

        # Line chart
        image = Image.open("utils/Images/guide_step5_1.png")
        image = image.resize((800, 800))
        st.image(image, caption="", use_column_width=False)

        st.write("➣ The app display the prediction results.")
        st.write(
            "➣ In the prediction results, the app display the most likely activity of the dog.")
        st.write("➣ The line chart of all actions over time for each data points.")

        # Bar chart
        image = Image.open("utils/Images/guide_step5_2.png")
        image = image.resize((800, 400))
        st.image(image, caption="", use_column_width=False)
        st.write(
            "➣ The app display the most occurred actions over a 5 different time intervals.")

        # Countplot
        image = Image.open("utils/Images/guide_step5_3.png")
        image = image.resize((800, 400))
        st.image(image, caption="", use_column_width=False)
        st.write("➣ The app display the distribution of all actions.")

    elif main_menu == "Application":

        # model_77 = keras.models.load_model('utils/model77.h5')
        # model_78 = keras.models.load_model('utils/model78.h5')
        # model_78_2 = keras.models.load_model('utils/model78_2.h5')
        # model_79 = keras.models.load_model('utils/model79.h5')
        model_80 = keras.models.load_model('utils/model80.h5')

        # models = [model_79, model_80]

        st.title("Upload the csv file here")
        smaple_csv_glink = "https://drive.google.com/file/d/1HB3OHQ-vVlR-AA2erIBRCukgI28BhDP2/view?usp=drive_link"
        st.write(
            f"➣ If you don't have a csv file, you can download a sample csv file from [here]({smaple_csv_glink}).")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            render_graphs = False

            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a string based IO:
            data = bytes_data.decode("utf-8")

            # Processing and loading spinner
            with st.spinner("Processing data and making predictions..."):
                processed_df = process_raw_csv_data(data)
                processed_data = test_data_preprocessing(processed_df)
                features_reshaped = reshape_features(processed_data)
                # all_actions, most_occurred_pred = ensemble_voting(
                #     models, features_reshaped)
                all_actions, most_occurred_pred = make_prediction(
                    model_80, features_reshaped)
                render_graphs = True

            if render_graphs:
                st.success("Prediction ready!")
                text = f"From my observation, your Pino is currently {most_occurred_pred}."
                st.markdown(
                    f"<p style='text-align: center; color: #008080; font-size: 26px;'>{text}</p>", unsafe_allow_html=True)

                # Line chart of all actions
                st.write("")
                st.write("")
                st.write(
                    f"➣ Below is a line chart of all actions over time for {len(all_actions)} data points.")

                fix, ax = plt.subplots()
                plt.plot(all_actions, color='blue')
                plt.title('All actions')
                plt.xlabel('Time')
                plt.ylabel('Actions')
                st.pyplot(fix)

                # Bar chart of all actions
                st.write("")
                st.write("")
                st.write(
                    f"➣ Below is a bar chart of the most occurred over a 5 different time intervals. This chart shows the most occurred actions of the dog for each time interval. Each interval lenght is {len(all_actions)//5}.")

                all_actions_array = np.array(all_actions)
                divided_actions = np.array_split(all_actions_array, 5)

                most_occurred_actions = []
                for action in divided_actions:
                    counter = Counter(action)
                    most_occurred_action, most_occurred_count = counter.most_common(1)[
                        0]
                    most_occurred_actions.append(
                        (most_occurred_action, most_occurred_count))
                actions, counts = zip(*most_occurred_actions)

                fix2, ax2 = plt.subplots()
                ax2.bar(range(len(actions)), counts, tick_label=actions)
                plt.xlabel('Actions')
                plt.ylabel('Count')
                plt.title('Most occurred actions')

                for i, count in enumerate(counts):
                    ax2.text(i, count, str(count), ha='center', va='bottom')

                st.pyplot(fix2)

                # Countplot of all actions
                st.write("")
                st.write("")
                st.write(
                    f"➣ Below is a countplot of all actions. This chart shows the distribution of {len(all_actions)} actions of the dog.")

                fix3, ax3 = plt.subplots()
                sns.countplot(x=all_actions, palette='Set2')
                plt.title('Countplot of all actions')
                plt.xlabel('Actions')
                plt.ylabel('Count')

                for p in ax3.patches:
                    ax3.annotate(format(p.get_height(), '.0f'),
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center',
                                 xytext=(0, 10),
                                 textcoords='offset points')

                st.pyplot(fix3)


if __name__ == "__main__":
    main()
