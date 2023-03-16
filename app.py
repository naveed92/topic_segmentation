import streamlit as st
from src.utils import window, get_depths, get_local_maxima, compute_threshold, get_threshold_segments

st.write("loading ...")

import spacy
nlp = spacy.load('en_core_web_sm')

def print_list(lst):
    for e in lst:
        st.markdown("- " + e)

# Demo start

st.subheader("Topic Segmentation Demo")

uploaded_file = st.file_uploader("choose a text file", type=["txt"])

if uploaded_file is not None: 
    st.session_state["text"] = uploaded_file.getvalue().decode('utf-8')

st.write("OR")

input_text = st.text_area(
    label="Enter text separated by newlines",
    value="",
    key="text",
    height=150

)

button=st.button('Get Segments')

# Radio bar
# BERT or TOPIC
select_names = ["LDA Topic", "BERT"]
model = st.radio(label='Select model', options=select_names, index=0)

if (button==True) and input_text != "":

    # Parse sample document and break it into sentences
    texts = input_text.split('\n')
    sents = []
    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            sents.append(sent)

    # Select tokens while ignoring punctuations and stopwords, and lowercase them
    MIN_LENGTH = 3
    tokenized_sents = [[token.lemma_.lower() for token in sent if 
                        not token.is_stop and not token.is_punct and token.text.strip() and len(token) >= MIN_LENGTH] 
                        for sent in sents]


    st.write("building topic model ...")

    # Build gensim dictionary and topic model
    from gensim import corpora, models
    import numpy as np

    np.random.seed(123)

    N_TOPICS = 5
    N_PASSES = 5

    dictionary = corpora.Dictionary(tokenized_sents)
    bow = [dictionary.doc2bow(sent) for sent in tokenized_sents]
    topic_model = models.LdaModel(corpus=bow, id2word=dictionary, num_topics=N_TOPICS, passes=N_PASSES)

    ###st.write(topic_model.show_topics())


    st.write("inferring topics ...")
    # Infer topics with minimum threshold
    THRESHOLD = 0.05
    doc_topics = list(topic_model.get_document_topics(bow, minimum_probability=THRESHOLD))

    # st.write(doc_topics)

    # get top k topics for each sentence
    k = 3
    top_k_topics = [[t[0] for t in sorted(sent_topics, key=lambda x: x[1], reverse=True)][:k] 
                    for sent_topics in doc_topics]
    # st.write(top_k_topics)

    ###st.write("apply window")

    from itertools import chain

    WINDOW_SIZE = 3

    window_topics = window(top_k_topics, n=WINDOW_SIZE)
    # assert(len(window_topics) == (len(tokenized_sents) - WINDOW_SIZE + 1))
    window_topics = [list(set(chain.from_iterable(window))) for window in window_topics]

    # Encode topics for similarity computation

    from sklearn.preprocessing import MultiLabelBinarizer

    binarizer = MultiLabelBinarizer(classes=range(N_TOPICS))

    encoded_topic = binarizer.fit_transform(window_topics)

    # Get similarities

    st.write("generating segments ...")

    from sklearn.metrics.pairwise import cosine_similarity

    sims_topic = [cosine_similarity([pair[0]], [pair[1]])[0][0] for pair in zip(encoded_topic, encoded_topic[1:])]
    # plot

    # Compute depth scores
    depths_topic = get_depths(sims_topic)
    # plot

    # Get local maxima
    filtered_topic = get_local_maxima(depths_topic, order=1)
    # plot

    ###st.write("compute threshold")
    # Automatic threshold computation
    # threshold_topic = compute_threshold(depths_topic)
    threshold_topic = compute_threshold(filtered_topic)

    # topk_segments = get_topk_segments(filtered_topic, k=5)
    # Select segments based on threshold
    threshold_segments_topic = get_threshold_segments(filtered_topic, threshold_topic)

    # st.write(threshold_topic)

    ###st.write("compute segments")

    segment_ids = threshold_segments_topic + WINDOW_SIZE

    segment_ids = [0] + segment_ids.tolist() + [len(sents)]
    slices = list(zip(segment_ids[:-1], segment_ids[1:]))

    segmented = [sents[s[0]: s[1]] for s in slices]

    for segment in segmented[:-1]:
        print_list([s.text for s in segment])
        st.markdown("""---""")
    print_list([s.text for s in segmented[-1]])

# TODO: Visualizations for topic, text
# TODO: BERT model switch
# TODO: Default input
# TODO: Pre segmented sentences
# TODO: Progress bar