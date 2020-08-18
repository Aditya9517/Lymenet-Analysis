from gensim import corpora, models
import pyLDAvis.gensim


if __name__ == '__main__':

    # Medical Question Corpus (file ending with ".mm")
    medical_question_corpus = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/Apr23Results/medicalQuestion.mm"

    # Medical Question Dictionary (file ending with ".dict")
    medical_question_dictionary = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/Apr23Results/medicalQuestion.dict"

    # LDA MODEL PATH
    lda24 = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/Apr23Results/generatedFiles_medicalQuestion/LDA_MODEL_24"

    # Loading the LDA model
    lda = models.LdaModel.load(lda24)

    # Loading the corpus
    corpus = corpora.MmCorpus(medical_question_corpus)

    # Loading the dictionary
    dictionary = corpora.Dictionary.load(medical_question_dictionary)

    # Topic Model Visualization using pyLDAvis

    # While using Google colab or Jupyter Notebooks enable this
    # pyLDAvis.enable_notebook()
    # pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='mmds')

    # Store visualization in a variable
    visualization = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='mmds')

    # Save topic model visualization as a HTML file
    pyLDAvis.save_html(visualization, 'LDA_Visualization.html')
