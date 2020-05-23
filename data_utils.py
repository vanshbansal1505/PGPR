from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random


class AmazonDataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

    def _load_file(self, filename): #returns list of strings
        with gzip.open(self.data_dir + filename, 'r') as f:
            # In Python 3, must use decode() to convert bytes to string!
            return [line.decode('utf-8').strip() for line in f] #.strip() removes trailing and leading spaces (and specified characters, if any)

    def load_entities(self):
        """Load 6 global entities from data files:
        `user`, `product`, `word`, `related_product`, `brand`, `category`.
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        entity_files = edict(
                user='users.txt.gz',
                product='product.txt.gz',
                word='vocab.txt.gz',
                related_product='related_product.txt.gz',
                brand='brand.txt.gz',
                category='category.txt.gz',
        )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab))) #vocab is key for vocab(list of str), vocab_size is key for len(vocab)
            print('Load', name, 'of size', len(vocab))
#setattr does self.name = third argument, so self.user, self.product etc. are initialized as dicts

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all reviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `word_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])
        product_distrib = np.zeros(self.product.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
            review_data.append((user_idx, product_idx, word_indices))
            product_distrib[product_idx] += 1 #increment number of reviews for product
            for wi in word_indices:
                word_distrib[wi] += 1 #increment number of words used for reviews
            word_count += len(word_indices)
        self.review = edict(
                data=review_data, #list of three membered tuples
                size=len(review_data), #number of reviews
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_product_relations(self):
        """Load 5 product -> ? relations:
        - `produced_by`: product -> brand,
        - `belongs_to`: product -> category,
        - `also_bought`: product -> related_product,
        - `also_viewed`: product -> related_product,
        - `bought_together`: product -> related_product,
        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict(
                produced_by=('brand_p_b.txt.gz', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt.gz', self.category),
                also_bought=('also_bought_p_p.txt.gz', self.related_product),
                also_viewed=('also_viewed_p_p.txt.gz', self.related_product),
                bought_together=('bought_together_p_p.txt.gz', self.related_product),
        )
        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict( #5 such dicts are made
                    data=[], #list of lists, see line 122
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = [] #list of entity tail indices
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0: #x takes values from 0 to number of brands - 1 ..
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1 #increase brand (x+1)'s entities by 1
                relation.data.append(knowledge) #hence, data is a list of lists
            setattr(self, name, relation) #self.produced_by ..
            print('Load', name, 'of size', len(relation.data))

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue #(i+1)th word never used for reviews
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)


class AmazonDataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset #dataset object
        self.batch_size = batch_size #default is 64
        self.review_size = self.dataset.review.size #number of reviews, dtype = int
        self.product_relations = ['produced_by', 'belongs_to', 'also_bought', 'also_viewed', 'bought_together'] #load 5 relations
        self.finished_word_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size) #randomly permutes np.arange(self.review_size)
        #np.arange(x) returns ndarray of evenly spaced numbers in [0, x)
        self.cur_review_i = 0 #initialize current review index
        self.cur_word_i = 0 #initialize current word index
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        #(user index, product index, word indices, brand index, category index, related product index)
        batch = []
        review_idx = self.review_seq[self.cur_review_i]
        user_idx, product_idx, text_list = self.dataset.review.data[review_idx] #unpack
        product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations} #maps pr to list in data at index = product_idx

        while len(batch) < self.batch_size:
            # 1) Sample the word
            word_idx = text_list[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]: #random.random() returns a random floating point in [0,1)
                data = [user_idx, product_idx, word_idx]
                for pr in self.product_relations:
                    if len(product_knowledge[pr]) <= 0: #if list is empty, put -1
                        data.append(-1)
                    else:
                        data.append(random.choice(product_knowledge[pr])) #returns random element from input sequence
                batch.append(data)

            # 2) Move to next word/review
            self.cur_word_i += 1 #move to next word
            self.finished_word_num += 1
            if self.cur_word_i >= len(text_list): #exhausted all words in this list
                self.cur_review_i += 1 #move to next review
                if self.cur_review_i >= self.review_size: #all reviews exhausted
                    self._has_next = False #batch is ready
                    break
                self.cur_word_i = 0 #reset counter
                review_idx = self.review_seq[self.cur_review_i]
                user_idx, product_idx, text_list = self.dataset.review.data[review_idx]
                product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}

        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next
