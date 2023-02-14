import streamlit as st
import matplotlib.pyplot as plt
from sthelper import StHelper
import data_helper

# import all datasets
concentric,linear,outlier,spiral,ushape,xor = data_helper.load_dataset()

# configure matplotlib styling
plt.style.use('seaborn-bright')

# Dataset selection dropdown
st.sidebar.markdown("# Voting Classifier")
dataset = st.sidebar.selectbox(
    "Dataset",
    ("U-Shaped", "Linearly Separable", "Outlier","Two Spirals","Concentric Circles","XOR")
)

# Estimator Multi-select
estimators = st.sidebar.multiselect(
    'Estimators',
    [
        'KNN',
        'Logistic Regression',
        'Gaussian Naive Bayes',
        'SVM',
        'Random Forest'
    ]
)

# Voting type radio button
voting_type = st.sidebar.radio(
    "Voting Type",
    (
        'hard',
        'soft',
    )
)



st.header(dataset)
fig, ax = plt.subplots()

# Plot initial graph
df = data_helper.load_initial_graph(dataset,ax)
orig = st.pyplot(fig)

# Extract X and Y
X = df.iloc[:,:2].values
y = df.iloc[:,-1].values

# Create sthelper object
sthelper = StHelper(X,y)

# On button click
if st.sidebar.button("Run Algorithm"):
    algos = sthelper.create_base_estimators(estimators,voting_type)
    voting_clf,voting_clf_accuracy = sthelper.train_voting_classifier(algos,voting_type)
    sthelper.draw_main_graph(voting_clf,ax)
    orig.pyplot(fig)
    figs = sthelper.plot_other_graphs(algos)
    # plot accuracies


    st.sidebar.header("Classification Metrics")
    st.sidebar.text("Voting Classifier accuracy:" + str(voting_clf_accuracy))

    accuracies = sthelper.calculate_base_model_accuracy(algos)

    for i in range(len(accuracies)):
        st.sidebar.text("Accuracy for Model " + str(i+1) + " - " + str(accuracies[i]))

    counter = 0
    for i in st.beta_columns(len(figs)):
        with i:
            st.pyplot(figs[counter])
            st.text(counter)
        counter+=1