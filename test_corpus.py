import unittest
import corpus


class TestCorpus(unittest.TestCase):

    def setUp(self):
        self.corpus = corpus.Corpus()
        self.size = 100
        for index in range(self.size):
            self.corpus.add_instance([index], [str(index)])
        self.assertEqual(len(self.corpus), self.size)

    def test_split_corpus(self):
        """Splits a corpus in three partial parts."""
        partitions = [10, 20, 50]
        splited_corpus = self.corpus.split(partitions)
        self.assertEqual(len(partitions), len(splited_corpus))
        for index, corpus in enumerate(splited_corpus):
            for instance, target in zip(corpus.instances,
                                        corpus.primary_targets):
                self.assertEqual((1, 1), instance.shape)
                # Check the targets still corresponds to the instances
                self.assertEqual(str(instance.toarray()[0][0]), target)

    def test_split_corpus_bigger(self):
        """The sums of the parts is bigger than the corpus."""
        partitions = [10, 90, 10]
        splited_corpus = self.corpus.split(partitions)
        self.assertIsNone(splited_corpus)


if __name__ == '__main__':
    unittest.main()