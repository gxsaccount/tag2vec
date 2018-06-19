/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.doc2vec;

import commonclasses.TaggedDocument;
//import commonclasses.TaggedSentence;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

//import weka.core.Instances;
//import weka.core.converters.CSVLoader;
/**
 *
 * @author I353540
 */
public class Doc2Vec {

    static Logger logger = Logger.getLogger(Doc2Vec.class.getName());
    private List<Float> cum_table;
    private Map<String, List<Long>> vocab;
    private List<String> newVocab;
    private List<String> index2word;
    private Map<String, List<Long>> docvecs;
    private List<String> offset2doctag;
    private int n_newWords;
    private int n_newTags;
    private boolean learn_words;

    private boolean save_model;
    private int negative;
    private int window;
    private int batch_words;
    private int max_vocab_size;
    private int corpus_count;
    private double sample;
    private int vector_size;
    private int layer1_size;
    private String save_path;
    private String modelPath;
    private Session sess;
    private List<TaggedDocument> train_corpus;
    private Tensor exist_vocabsTensor;
    private Tensor exist_doc2vecsTensor;
    private Tensor exit_nce_weightsTensor;
    private String dataPath;

    public Doc2Vec() {
        this.epochs = 1;
        this.dataPath = "tag_copus.txt";
        this.save_path = "model\\variables\\variables";
        this.modelPath = "model";
        this.layer1_size = 300;
        this.vector_size = 300;
        this.sample = 1e-05;
        this.corpus_count = 0;
        this.max_vocab_size = 10000000;
        this.batch_words = 200;
        this.window = 1;
        this.negative = 5;
        this.save_model = false;
        this.learn_words = false;
        this.n_newWords = 0;
        this.n_newTags = 0;
        this.offset2doctag = new ArrayList<>();
        this.docvecs = new HashMap<>();
        this.index2word = new ArrayList<>();
        this.vocab = new HashMap<>();
        this.cum_table = new ArrayList<>();
    }

    public Doc2Vec(Map<String, String> params) {
        this();
        this.epochs = Integer.parseInt(params.get("epochs"));
        this.dataPath = params.get("dataPath");
        this.save_path = params.get("savePath");
        this.n_newWords = 0;
        this.n_newTags = 0;
        this.learn_words = Boolean.parseBoolean(params.get("learn_words"));
        this.save_model = Boolean.parseBoolean(params.get("save_model"));
        this.negative = Integer.parseInt(params.get("negative"));
        this.window = Integer.parseInt(params.get("window"));
        this.batch_words = Integer.parseInt(params.get("batch_words"));
        this.max_vocab_size = Integer.parseInt(params.get("max_vocab_size"));
        this.corpus_count = 0;
        this.sample = Double.parseDouble(params.get("sample"));
        this.layer1_size = Integer.parseInt(params.get("layer1_size"));
        this.modelPath = params.get("modelPath");
        this.offset2doctag = new ArrayList<>();
        this.docvecs = new HashMap<>();
        this.index2word = new ArrayList<>();
        this.vocab = new HashMap<>();
        this.cum_table = new ArrayList<>();
    }

    public Doc2Vec(int n_newWords, int epochs, int n_newTags, boolean learn_words, boolean save_model, int negative, int window, int batch_words, int max_vocab_size, int corpus_count, double sample, int layer1_size, String modelPath, String savePath, String dataPath) {
        this.epochs = epochs;
        this.dataPath = dataPath;
        this.save_path = savePath;
        this.n_newWords = n_newWords;
        this.n_newTags = n_newTags;
        this.learn_words = learn_words;
        this.save_model = save_model;
        this.negative = negative;
        this.window = window;
        this.batch_words = batch_words;
        this.max_vocab_size = max_vocab_size;
        this.corpus_count = corpus_count;
        this.sample = sample;
        this.layer1_size = layer1_size;
        this.modelPath = modelPath;
        this.offset2doctag = new ArrayList<>();
        this.docvecs = new HashMap<>();
        this.index2word = new ArrayList<>();
        this.vocab = new HashMap<>();
        this.cum_table = new ArrayList<>();
    }

    public static void main(String[] args) throws UnsupportedEncodingException {
        Doc2Vec dv = new Doc2Vec();
        dv.loadAndTrain();
    }
    private int epochs;

    private void trainModel(Session s) {
        int epochs = 1;
        s.runner()
                .feed("n_newTags:0", Tensor.create(n_newTags))
                .feed("n_newWords:0", Tensor.create(n_newWords))
                .feed("wv_plh:0", exist_vocabsTensor)
                .feed("doc_vec:0", exist_doc2vecsTensor)
                .feed("sp_plh:0", exit_nce_weightsTensor)
                .addTarget("init")
                .run();
        for (int i = 0; i < epochs; i++) {
            float loss = 0;
            int times = 0;
            int batch_size = 0;
            int words_count = 0;
            for (int offset = 0; offset < train_corpus.size(); offset += batch_size) {
                batch_size = 0;
                for (int j = offset; j < train_corpus.size(); j++) {
                    int tmp = words_count + train_corpus.get(j).getWords().size();

                    if (tmp > this.batch_words) {
                        words_count = 0;
                        break;
                    } else {
                        batch_size++;
                        words_count = tmp;
                    }

                }
                if (learn_words) {
                    List<Long> inputs_sg = new ArrayList<>(), labels_sg = new ArrayList<>();
                    long[] inputs_sg_ = inputs_sg.stream().mapToLong(l -> l).toArray();
                    long[] labels_sg_ = labels_sg.stream().mapToLong(l -> l).toArray();
                    generate_batch_words(train_corpus, labels_sg, labels_sg, offset, batch_size);
                    if (inputs_sg.size() > 0 && labels_sg.size() > 0) {
                        List<Tensor<?>> reslut = s.runner()
                                .feed("inputs_sg:0", Tensor.create(inputs_sg_))
                                .feed("labels_sg:0", Tensor.create(labels_sg_))
                                .fetch("optimizer_sg")
                                .fetch("lose_sg:0").run();
                    }
                }
                List<Long> inputs_db = new ArrayList<>();

                List<List<Long>> labels_db = new ArrayList<>();

                generate_label(train_corpus, inputs_db, labels_db, offset, batch_size);
                long[] inputs_db_ = inputs_db.stream().mapToLong(l -> l).toArray();
                long[][] labels_db_ = new long[labels_db.size()][];
                for (int j = 0; j < labels_db.size(); j++) {
                    labels_db_[j] = labels_db.get(j).stream().mapToLong(l -> l).toArray();
                }
                List<Tensor<?>> resluts = s.runner()
                        .feed("inputs_db:0", Tensor.create(inputs_db_))
                        .feed("labels_db:0", Tensor.create(labels_db_))
                        .addTarget("optimizer_db")
                        .fetch("lose_db:0")
                        .fetch("d2v_embeding:0")
                        .fetch("nce_weights:0")
                        .run();
                loss += resluts.get(0).floatValue();
                ++times;
            }
            logger.log(Level.INFO, "******************* loss : {0} at epoch {1} ********************************", new Object[]{loss / times, i});
        }

        s.runner().addTarget("copy_vocabs").addTarget("copy_d2v").addTarget("copy_nce");
    }
    Tensor oldWordTensor;

    public boolean loadModel() {
        String modelFile = this.modelPath;
        String dataFile = this.dataPath;
        boolean succ = false;
        String[] tags = {"d2v_tf", "train"};
        SavedModelBundle model = SavedModelBundle.load(modelFile, tags);

        sess = model.session();
        try {
            List<Tensor<?>> results = sess.runner()
                    .fetch("cum_table:0")
                    .fetch("words:0")
                    .fetch("words_index:0")
                    .fetch("doctags:0")
                    .fetch("doc_index:0")
                    .fetch("exist_vocabs:0")
                    .fetch("exist_doc2vec:0")
                    .fetch("exit_nce_weights:0")
                    .run();
            Iterator it = results.iterator();

            Tensor cumTable = (Tensor) it.next();
            float[] cumTableFloats = new float[(int) cumTable.shape()[0]];
            cumTable.copyTo(cumTableFloats);
            for (float cumTableFloat : cumTableFloats) {
                cum_table.add(cumTableFloat);
            }

            oldWordTensor = (Tensor) it.next();
            byte[][] wordsBytes = new byte[(int) oldWordTensor.shape()[0]][];
            oldWordTensor.copyTo(wordsBytes);
            for (byte[] wordseByte : wordsBytes) {
                this.index2word.add(new String(wordseByte, "UTF-8"));
            }

            List<List<Long>> wordIndex = new ArrayList<>();
            Tensor<?> wordsIndexTensor = (Tensor) it.next();
            LongBuffer wordsIndexBuf = LongBuffer.allocate(wordsIndexTensor.numElements());
            wordsIndexTensor.writeTo(wordsIndexBuf);
            long[] wordsIndexArr = wordsIndexBuf.array();
            int batch = (int) wordsIndexTensor.shape()[1];
            ArrayList<Long> tmp;
            for (int i = 0; i < wordsIndexArr.length; i += batch) {
                tmp = new ArrayList<>();
                for (int j = i; j < i + batch; j++) {
                    tmp.add(wordsIndexArr[j]);
                }
                wordIndex.add(tmp);
            }

            for (int i = 0; i < index2word.size(); i++) {
                this.vocab.put(index2word.get(i), wordIndex.get(i));
            }
            Tensor doctagsTensor = (Tensor) it.next();
            byte[][] doctagsBytes = new byte[(int) doctagsTensor.shape()[0]][];
            doctagsTensor.copyTo(doctagsBytes);
            for (byte[] doctagsByte : doctagsBytes) {
                this.offset2doctag.add(new String(doctagsByte, "UTF-8"));
            }

            Tensor<?> doctagtoindexTensor = (Tensor) it.next();
            List<List<Long>> doctagtoindex = new ArrayList<>();
            long[][] doctagtoindexFloats = new long[(int) doctagtoindexTensor.shape()[0]][(int) doctagtoindexTensor.shape()[1]];
            doctagtoindexTensor.copyTo(doctagtoindexFloats);
            for (long[] tmpArr : doctagtoindexFloats) {
                ArrayList<Long> temp = new ArrayList<>();
                for (long g : tmpArr) {
                    temp.add(g);
                }
                doctagtoindex.add(temp);
            }

            for (int i = 0; i < offset2doctag.size(); i++) {
                docvecs.put(offset2doctag.get(i), doctagtoindex.get(i));
            }

            List<List<Float>> exist_vocabs = new ArrayList<>();
            exist_vocabsTensor = (Tensor) it.next();
            List<List<Float>> exist_doc2vec = new ArrayList<>();
            exist_doc2vecsTensor = (Tensor) it.next();
            List<List<Float>> exit_nce_weights = new ArrayList<>();
            exit_nce_weightsTensor = (Tensor) it.next();
            train_corpus = loadCorpus(dataFile);
            build_vocab(train_corpus, 1);
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(Doc2Vec.class.getName()).log(Level.SEVERE, null, ex);
        }
        return succ;
    }

    private void generate_label(List<TaggedDocument> sentences, List<Long> batch, List<List<Long>> label, int offset, int batchSize) {
//        System.out.println(batchSize);
        for (int i = offset; i < Math.min(offset + batchSize, sentences.size()); i++) {
            if (i > sentences.size()) {
                break;
            }
            TaggedDocument sentence = sentences.get(i);
            List<Long> doctag_indexes = new ArrayList<>();
            for (String tag : sentence.getTags()) {
                List<Long> index = this.docvecs.get(tag);
                if (index != null) {
                    doctag_indexes.add(index.get(0));
                }
            }
            for (String word : sentence.getWords()) {
                List<Long> predict_word = vocab.get(word);
                List<Long> word_indices = _random_sample_negative(predict_word);
                batch.addAll(doctag_indexes);
                for (int j = 0; j < doctag_indexes.size(); j++) {
                    label.add(word_indices);
                }
            }
        }

    }

    public void loadAndTrain() {

        loadModel();
        trainModel(sess);
       if (save_model) {
        saveModel(sess);
       }

    }

//    public long nextLong(Random rng, long n)
    private List<Long> _random_sample_negative(List<Long> predict_word) {
        List<Long> word_indices = new ArrayList();
        word_indices.add(predict_word.get(0));
        Random ran = new Random();
        while (word_indices.size() < negative + 1) {
            //cum_table 是一个升序序列
            float tmp = ran.nextInt();
            int index = Collections.binarySearch(this.cum_table, tmp);
            Long w = new Long(index < 0 ? -index : index);
            if (!w.equals(predict_word.get(0))) {
                word_indices.add(w);
            }
        }
        return word_indices;
    }

    private void generate_batch_words(List<TaggedDocument> sentences, List<Long> batch, List<Long> label, int offset, int batchSize) {
        for (int i = offset; i < Math.min(offset + batchSize, sentences.size()); i++) {
            if (i > sentences.size()) {
                break;
            }
            TaggedDocument sentence = sentences.get(i);
            List<List<Long>> word_vocabs = new ArrayList<>();
            for (String word : sentence.getWords()) {
                List<Long> index = this.docvecs.get(word);
                Random ran = new Random();
                if (index.get(2) > ran.nextInt()) {
                    word_vocabs.add(index);
                }
            }
            for (int pos = 0; pos < word_vocabs.size(); pos++) {
                List<Long> word = word_vocabs.get(pos);
                int start = Math.max(0, pos - window);
                for (int pos2 = start; pos2 < pos + window + 1; pos2++) {
                    List<Long> word2 = word_vocabs.get(pos2);
                    if (pos2 != pos) {
                        batch.add(word.get(0));
                        List<Long> word_index = _random_sample_negative(word2);
                        label.addAll(word_index);
                    }
                }
            }
        }

    }
//Apache Commons IO流
//    private List<TaggedDocument> loadCorpusLineByLine() {
//        LineIterator it = FileUtils.lineIterator(theFile, UTF - 8);
//        try {
//            while (it.hasNext()) {
//                String line = it.nextLine();
//                // do something with line 
//            }
//        } finally {
//            LineIterator.closeQuietly(it);
//        }
//    }

    //文件过大未处理：一行一行读取
    //
    private List<TaggedDocument> loadCorpus(String corpus) {
        List<TaggedDocument> documents = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(new File(corpus).toPath());
            for (String line : lines) {
                String[] temp = line.split("#");
                String[] tags = temp[0].split(",");
                String[] words = temp[1].split(",");

                List<String> tagsList = new ArrayList<>();
                List<String> wordsList = new ArrayList<>();
                for (String tag : tags) {
                    if (tag != "") {
                        tagsList.add(tag);
                    }
                }
                for (String word : words) {
                    if (word != "") {
                        wordsList.add(word);
                    }
                }
                documents.add(new TaggedDocument(tagsList, wordsList));
            }
        } catch (IOException ex) {
            Logger.getLogger(Doc2Vec.class.getName()).log(Level.SEVERE, null, ex);
        }
//    ???
//        corpus_count = documents.size();
        return documents;
    }

    private void build_vocab(List<TaggedDocument> documents, int min_count) {
        scan_tags(documents);
        Map<String, Integer> vocabs = scan_vocab(documents);
        scale_vocab(vocabs, min_count, false);
        finalize_vocab("\\0");
    }

    private void scan_tags(List<TaggedDocument> documents) {
        List<String> new_tags = new ArrayList<>();
        for (TaggedDocument document : documents) {
            int documnetLength = document.getWords().size();
            for (String tag : document.getTags()) {
                if (!docvecs.containsKey(tag)) {
                    new_tags.add(tag);
                    List<Long> list = new ArrayList<>();
                    list.add((long) docvecs.size());
                    list.add((long) documnetLength);
                    docvecs.put(tag, list);
                    offset2doctag.add(tag);
                } else {
                    docvecs.get(tag).set(1, docvecs.get(tag).get(1) + documnetLength);
                }
            }
        }
        n_newTags = new_tags.size();
    }

    private Map<String, Integer> scan_vocab(List<TaggedDocument> documents) {
        int min_reduce = 1;
        Map<String, Integer> vocab = new HashMap<>();//统计词频
        for (TaggedDocument document : documents) {
            //词频
            for (String word : document.getWords()) {
                if (!vocab.containsKey(word)) {
                    vocab.put(word, 1);
                } else {
                    vocab.put(word, vocab.get(word) + 1);
                }
            }
            //内存处理留坑
            if (this.max_vocab_size != 0 && this.vocab.size() > this.max_vocab_size) {
                prune_vocab(vocab, min_reduce, "");
            }
//            utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
        }

        return vocab;
    }

    private void prune_vocab(Map<String, Integer> vocab, int min_reduce, String trim_rule) {
        int result = 0;
        int old_len = vocab.size();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();
            if (!keep_vocab_item(key, value, min_reduce, trim_rule)) {
                vocab.remove(key);
            }
        }
        logger.log(Level.INFO, "pruned out {0}with count <= {1}(before {2}, after {3}", new Object[]{old_len - vocab.size(), old_len, old_len, vocab.size()});
    }

    private boolean keep_vocab_item(String word, int count, int min_count, String trim_rule) {
        return count >= min_count;
    }

    //统计新词
    private void scale_vocab(Map<String, Integer> vocabs, int min_count, boolean dry_run) {
        double sample = this.sample;
        int drop_unique = 0, drop_total = 0;
        int new_total = 0, pre_exist_total = 0;
        List<String> new_words = new ArrayList<>();
        List<String> pre_exist_words = new ArrayList<>();
        int n_words_exit = vocab.size();
        for (Map.Entry<String, Integer> entry : vocabs.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();
            if (value >= min_count) {
                if (this.vocab.containsKey(key)) {
                    pre_exist_words.add(key);
                    pre_exist_total += value;
                    if (!dry_run) {
                        this.vocab.get(key).set(1, vocab.get(key).get(1) + value);
                    }
                } else {
                    new_words.add(key);
                    new_total += value;
                    if (!dry_run) {
                        List<Long> tmp = new ArrayList<>();
                        tmp.add(new Long(this.index2word.size()));
                        tmp.add(new Long(value));
                        tmp.add(0l);
                        this.vocab.put(key, tmp);
                        this.index2word.add(key);
                    }
                }
            } else {
                drop_unique += 1;
                drop_total += value;
            }
        }
        int n_newWords = new_words.size();
        logger.log(Level.INFO, "n_words_exit + n_newWords == vocabs size : {0}", n_words_exit + n_newWords == this.vocab.size());
        System.out.println("com.mycompany.doc2vec.Doc2Vec.scale_vocab() " + n_newWords);
        this.newVocab = new_words;
        this.n_newWords = n_newWords;
        ArrayList<String> retain_words = new ArrayList<>();
        retain_words.addAll(new_words);
        retain_words.addAll(pre_exist_words);
        int retain_total = new_total + pre_exist_total;
        double threshold_count = -1;

        if (sample >= 1.0) {
            // no words downsampled
            threshold_count = retain_total;
        } else if (sample < 1.0) {
            //traditional meaning : set parameter as proportion of total
            threshold_count = sample * retain_total;
        } else {
            //new shorthand: sample >= 1 means downsample all words with higher count than sample 
            threshold_count = sample * (3 + Math.sqrt(5)) / 2;
        }
        int downsample_total = 0, downsample_unique = 0;
        for (String w : retain_words) {
            Integer v = vocabs.get(w);
            //(sqrt(v / threshold_count) + 1) * (threshold_count / v)
            double word_probability = Math.sqrt((v / threshold_count) + 1) * (threshold_count / v);

            if (word_probability < 1.0) {
                downsample_unique += 1;
                downsample_total += word_probability * v;
            } else {
                word_probability = 1.0;
                downsample_total += v;
            }
            if (!dry_run) {
                this.vocab.get(w).set(2, Math.round(Math.pow(word_probability * 2, 32)));

            }
        }
    }
//
    //

    private void drawDiagram() {
        throw new UnsupportedOperationException("not support yet..");
//        variables.docvec_syn0 = exist_doc2vec.eval()
//    doc2vec = getVec(variables, doctag)
//    drawDiagram(doc2vec, docs, label='productName', bounds_x=None, bounds_y=None)
    }

    private void finalize_vocab(String null_word) {
        //Build tables and model weights based on  vocabulary settings.
        double power = 0.75;
        double domain = Math.pow(2, 31) - 1;
        if (this.negative != 0) {
            //build the table for drawing random words (for negative sampling)
            int vocab_size = this.index2word.size();
            this.cum_table = new ArrayList<>(Collections.nCopies(vocab_size, 0.0f));
//            Collections.fill(cum_table, new Double(0));
            //# compute sum of all power (Z in paper)\
            double train_words_pow = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                // train_words_pow += model.vocab[model.index2word[word_index]][1] ** power;
                String j = this.index2word.get(i);
                Long k = this.vocab.get(j).get(1);
                train_words_pow += Math.pow(k, power);
            }
            double cumulative = 0.0;
            for (int i = 0; i < vocab_size; i++) {
//            model.cum_table[word_index] = round(cumulative / train_words_pow * domain)
                cumulative += Math.pow(this.vocab.get(this.index2word.get(i)).get(1), power);
                //强转可能有坑！！！
                this.cum_table.set(i, (float) Math.round(cumulative / train_words_pow * domain));
            }
        }
        if ((!this.vocab.containsKey(null_word)) && (!this.vocab.containsKey("\\0"))) {
//            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
//        # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            // word, v = '\0', [len(model.vocab), 1, 0]
            this.index2word.add("\\0");
            List<Long> tmp = new ArrayList<>();
            tmp.add(new Long(this.vocab.size()));
            tmp.add(1l);
            tmp.add(0l);
            this.vocab.put("\\0", tmp);
        }
    }

    public void saveModel(Session sess) {
        FloatBuffer cumTable = FloatBuffer.allocate(cum_table.size());
        long[] cumTableShape = new long[]{cum_table.size()};
        for (float double1 : cum_table) {
            cumTable.put(double1);
        }

        LongBuffer words_index = LongBuffer.allocate(vocab.size() * 3);
        long[] words_indexShap = new long[]{vocab.size(), 3};
        for (Map.Entry<String, List<Long>> entry : vocab.entrySet()) {
            String key = entry.getKey();
            List<Long> value = entry.getValue();
            for (Long long1 : value) {
                words_index.put(long1);
            }
        }
        //words
        int i = 0;
        byte[][] new_words = new byte[this.n_newWords][];
        for (String string : this.newVocab) {
            new_words[i] = string.getBytes(Charset.forName("UTF-8"));
            i++;
        }
        Tensor words;
        if (this.n_newWords > 0) {
            words = TensorflowUtil.concat(oldWordTensor, Tensor.create(new_words), 0);
        } else {
            words = oldWordTensor;
        }
        byte[][] doctags = new byte[docvecs.size()][];
        LongBuffer doctags_index = LongBuffer.allocate(docvecs.size() * 2);
        long[] doctags_indexShape = new long[]{docvecs.size(), 2};
        i = 0;
        for (Map.Entry<String, List<Long>> entry : docvecs.entrySet()) {
            String key = entry.getKey();
            List<Long> value = entry.getValue();
            doctags[i] = key.getBytes(Charset.forName("UTF-8"));
            for (Long long1 : value) {
                doctags_index.put(long1);
            }
            i++;
        }
        cumTable.flip();
        doctags_index.flip();
        words_index.flip();

        sess.runner().feed("ct_plh:0", Tensor.create(cumTableShape, cumTable))
                .feed("index2w_plh:0", words)
                .feed("words_index_plh:0", Tensor.create(words_indexShap, words_index))
                .feed("doctags_plh:0", Tensor.create(doctags))
                .feed("doc_index_plh:0", Tensor.create(doctags_indexShape, doctags_index))
                .addTarget("save_dict_init").run();
        byte[] path = this.save_path.getBytes(Charset.forName("UTF-8"));
        
        sess.runner().feed("save/Const:0", Tensor.create(path))
                .addTarget("save/Identity").run();
    }

    private List<TaggedDocument> getBatchSentences(List<TaggedDocument> sentences, Integer offset) {
        int batch_size = 0;
        int start = 0;
        for (int i = offset; i < sentences.size(); i++) {
            batch_size += sentences.get(i).getWords().size();
            if (batch_size < this.batch_words) {
                offset++;
            } else {
                start = offset++;
                Class<?> cla = Integer.class;
                try {
                    Field field = cla.getDeclaredField("num");
                    field.setAccessible(true);
                    field.set(offset, i);

                } catch (IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException ex) {
                    Logger.getLogger(Doc2Vec.class
                            .getName()).log(Level.SEVERE, null, ex);
                }
                break;
            }
        }

        return sentences.subList(start, offset);

    }
}

class BatchSenProducter implements Runnable {

    int batchSize;
    int offset;
    Queue<List<TaggedDocument>> batchSen;
    List<TaggedDocument> sentences;

    public BatchSenProducter(int batchSize, int offset, Queue<List<TaggedDocument>> batchSen, List<TaggedDocument> sentences) {
        this.batchSize = batchSize;
        this.offset = offset;
        this.batchSen = batchSen;
        this.sentences = sentences;
    }

    @Override
    public void run() {
        int start = batchSize * offset;
        try {
            synchronized (batchSen) {
                while (start < sentences.size() && batchSen.isEmpty()) {
                    wait();
                }
                batchSen.add(new ArrayList<>(sentences.subList(start, Math.max(start + batchSize, sentences.size()))));
                offset++;
                notifyAll();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}

class BatchSenGeter implements Callable<Boolean> {

    Queue<List<TaggedDocument>> batchSen;

    public BatchSenGeter(Queue<List<TaggedDocument>> batchSen) {
        this.batchSen = batchSen;
    }

    @Override
    public Boolean call() throws Exception {
        synchronized (batchSen) {
            while (batchSen.isEmpty()) {
                wait();
            }

        }
        return true;
    }

}
