package com.ansj.vec;
import syllabify.model.English;
import syllabify.model.ISyllable;
import syllabify.model.Word;

import java.io.BufferedReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import com.ansj.vec.util.MapCount;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.PinYin;
import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;
public class Serial {

	private Map<String, Neuron> wordMap = new HashMap<String, Neuron>(); 
	private Map<String, PinYin> pinyinMap = new HashMap<>();
	
	
	/**
	 * 训练多少
	 */
	private int layerSize = 100;

	/**
	 * 上下文窗口大小
	 */
	private int window = 5;
	private int negative = 5;

	private int min_reduce = 5;

	private double sample = 1e-3;
	private double alpha = 0.025;
	private double startingAlpha = alpha;
	private double wa = 1;
	private double wb = 1;

	public int EXP_TABLE_SIZE = 1000;
	private int table_size = (int) 1e8;

	private double[] expTable = new double[EXP_TABLE_SIZE];
	private String[] table = new String[table_size];

	private int trainWordsCount = 0;

	private int MAX_EXP = 6;

	private boolean pyfinish = false;

	public Serial(Boolean isCbow, Boolean ispinyin, Integer layerSize, Integer window, Double alpha, Double sample) {
		createExpTable();
		if (layerSize != null)
			this.layerSize = layerSize;
		if (window != null)
			this.window = window;
		if (alpha != null)
			this.alpha = alpha;
		if (sample != null)
			this.sample = sample;
	}

	public Serial() {
		createExpTable();
	}

	/**
	 * trainModel
	 * 
	 * @throws IOException
	 */
	private void trainModel(File file) throws IOException {
		for (int trainNum = 0; trainNum < 2; trainNum++) {
			try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
				String temp = null;
				long nextRandom = 5;
				int wordCount = 0;
				int lastWordCount = 0;
				int wordCountActual = 0;
				while ((temp = br.readLine()) != null) {
					if (wordCount - lastWordCount > 10000) {
						System.out.println("alpha:" + alpha + "\tProgress: "
								+ (int) (wordCountActual / (double) (trainWordsCount + 1) * 100) + "%");
						wordCountActual += wordCount - lastWordCount;
						lastWordCount = wordCount;
						alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
						if (alpha < startingAlpha * 0.0001) {
							alpha = startingAlpha * 0.0001;
						}
					}
					String[] strs = temp.split(" ");
					wordCount += strs.length;
					List<WordNeuron> sentence = new ArrayList<WordNeuron>();
					for (int i = 0; i < strs.length; i++) {
						if (wordMap.containsKey(strs[i])) {
							Neuron entry = wordMap.get(strs[i]);
							if (entry == null) {
								continue;
							}
							// The subsampling randomly discards frequent words
							if (sample > 0) {
								double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
										* (sample * trainWordsCount) / entry.freq;
								nextRandom = nextRandom * 25214903917L + 11;
								if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
									continue;
								}
							}
							sentence.add((WordNeuron) entry);
						}
					}

					for (int index = 0; index < sentence.size(); index++) {
						nextRandom = nextRandom * 25214903917L + 11;
						cbowGram(index, sentence, (int) nextRandom % window);
					}

				}
				br.close();
			}
			if (pyfinish == false) {//第一阶段的训练完成，将第一轮音向量作为第二轮训练的辅助词向量
				// set word embedding and temp embedding
				PinYin tempy;
				String pronun;
				double charsize;
				for (Entry<String, Neuron> word : wordMap.entrySet()) {
					WordNeuron wordNeuron = (WordNeuron) word.getValue();
					for (int i = 0; i < layerSize; i++) {
						// wordNeuron.syn0[i] = 0;
						wordNeuron.tempsyn0[i] = 0;
					}
					charsize = word.getKey().length();
					
					double count=0;
					double[] v=new double[100];
					Word w = new Word(word.getKey());
					English lang = new English();
					lang.nucleusPass(w);
					lang.onsetPass(w);
					for (ISyllable s : w.getSyllables()) {
						count++;
					}
					for (ISyllable s : w.getSyllables()) {
						String p=s.toString();
						tempy = pinyinMap.get(p);
						for (int k = 0; k < layerSize; k++) {
							v[k] =v[k]+tempy.syn0[k]/count;
						}
							
					}
					for (int k = 0; k < layerSize; k++) 
						wordNeuron.tempsyn0[k] =0.9*wordNeuron.tempsyn0[k]+0.1*v[k];
					
				}
			}
			pyfinish = true;
		}
		System.out.println("Vocab size: " + wordMap.size());
		System.out.println("Words in train file: " + trainWordsCount);
		System.out.println("sucess train over!");

	}

	/**
	 * 词袋模型
	 * 
	 * @param index
	 * @param sentence
	 * @param b
	 */
	private void cbowGram(int index, List<WordNeuron> sentence, int b) {
		WordNeuron word = sentence.get(index);
		int a, c = 0;
		double charsize;

		double[] neu1e = new double[layerSize]; // 形误差项
		double[] neu1 = new double[layerSize]; // 形表示
		double[] neu2 = new double[layerSize]; // 音表示
		WordNeuron last_word;
		PinYin tempy;
		String pronun;

		for (a = b; a < window * 2 + 1 - b; a++)//动态窗口：窗口大小是window-b
			if (a != window) {
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;

				charsize = last_word.name.length();
				if (pyfinish) {//处理输入层
					for (c = 0; c < layerSize; c++) {
						neu1[c] += last_word.syn0[c];
					}
				}
				
				
				double count=0;
				double[] v=new double[100];
				Word w = new Word(last_word.name);
				English lang = new English();
				lang.nucleusPass(w);
				lang.onsetPass(w);
				for (ISyllable s : w.getSyllables()) {
					count++;
				}
				for (ISyllable s : w.getSyllables()) {
					String p=s.toString();
					tempy = pinyinMap.get(p);
					for (int k = 0; k < layerSize; k++) {
						v[k] = v[k] + tempy.syn0[k]/count;
					}
				}
				for (int k = 0; k < layerSize; k++) {
					neu2[k] = 0.9*neu2[k] + 0.1*v[k];
				}
				
							
				

			}

		if (negative > 0) {
			String target;
			int label;
			int next_random;
			Random random = new Random();
			double f1, f2, g, q, f;
			WordNeuron targetNeuron = null;
			List<Pinyin> py;

			for (int d = 0; d < negative + 1; d++) {
				if (d == 0) {
					target = word.name;
					label = 1;
				} else {
					next_random = random.nextInt(table_size);
					target = table[next_random];
					if (target == word.name)
						continue;
					label = 0;
				}
				f1 = 0;
				f2 = 0;

				List<PinYin> pinyins = new ArrayList<PinYin>();
				double[] tempembedding = new double[layerSize];
				charsize = target.length();
				if (pyfinish) {//如果第一阶段音向量训练完成，tempembedding等于辅助词向量，即输出向量
					targetNeuron = (WordNeuron) wordMap.get(target);
					for (int i = 0; i < layerSize; i++) {
						tempembedding[i] = targetNeuron.tempsyn0[i];
					}
				} else {//否则还在训练的第一阶段，tempembedding等于辅助词向量
					
					double count=0;
					double[] v=new double[100];
					Word w = new Word(target);
					English lang = new English();
					lang.nucleusPass(w);
					lang.onsetPass(w);
					for (ISyllable s : w.getSyllables()) {
						count++;
					}
					for (ISyllable s : w.getSyllables()) {
						String p=s.toString();
						tempy = pinyinMap.get(p);
						pinyins.add(tempy);
						for (int k = 0; k < layerSize; k++) {
							v[k] = v[k] + tempy.tempsyn0[k]/count;//辅助音向量
						}
					}
					for (int k = 0; k < layerSize; k++) {
						tempembedding[k] = 0.9*tempembedding[k] + 0.1*v[k];//辅助音向量
					}
					
				}

				for (c = 0; c < layerSize; c++) {
					if (pyfinish) {
						f1 += neu1[c] * tempembedding[c];
					}
					f2 += neu2[c] * tempembedding[c];

				}
				f = f1 + f2;//（syn+pron）*tempembedding
				
				if (f > MAX_EXP) {
					q = 1;
				} else if (f < -MAX_EXP) {
					q = 0;
				} else {
					q = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				}
				
				
				g = alpha * (label - q);
				for (c = 0; c < layerSize; c++) {
					neu1e[c] += g * tempembedding[c];
				}

				if (pyfinish) {//处理输出层
					for (c = 0; c < layerSize; c++)
						targetNeuron.tempsyn0[c] += g * (neu1[c] + neu2[c]);
				} else {
					for (int i = 0; i < pinyins.size(); i++) {
						for (c = 0; c < layerSize; c++) {
							pinyins.get(i).tempsyn0[c] += g * neu2[c];
						}
					}
				}
			}
		}

		for (a = b; a < window * 2 + 1 - b; a++) {
			if (a != window) {
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;

				if (pyfinish) {
					for (c = 0; c < layerSize; c++)
						last_word.syn0[c] += neu1e[c];
				}

				charsize = last_word.name.length();

				double count=0;
				double[] v=new double[100];
				Word w = new Word(last_word.name);
				English lang = new English();
				lang.nucleusPass(w);
				lang.onsetPass(w);
				for (ISyllable s : w.getSyllables()) {
					count++;
				}
				for (ISyllable s : w.getSyllables()) {
					String p=s.toString();
					tempy = pinyinMap.get(p);
					for (int k = 0; k < layerSize; k++) {
						pinyinMap.get(p).syn0[k] = pinyinMap.get(p).syn0[k]/count + neu1e[k];
					}
				}
				
			}

		}
	}

	// reduce the vocabulary
	private void ReduceVocab() {
		List<String> r = new ArrayList<String>();
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			if (word.getValue().freq < min_reduce) {
				r.add(word.getKey());
			}
		}
		for (String string : r) {
			wordMap.remove(string);
		}
	}

	/**
	 * 统计词频
	 * 
	 * @param file
	 * @throws IOException
	 */
	private void readVocab(File file) throws IOException {
		MapCount<String> mc = new MapCount<>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			String temp = null;
			while ((temp = br.readLine()) != null) {
				String[] split = temp.split(" ");
				trainWordsCount += split.length;
				for (String string : split) {
					if (string.length() > 0 && !string.equals(" ")) {
						mc.add(string);
					}
				}
			}
		}
		for (Entry<String, Integer> element : mc.get().entrySet()) {
			WordNeuron tempw = new WordNeuron(element.getKey(), (double) element.getValue(), layerSize);
			wordMap.put(element.getKey(), tempw);
		}
	}

	private void getPinOrChar() {
		// add the pinyin to the pinyinMap
		
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			
			Word w = new Word(word.getKey());
			English lang = new English();
			lang.nucleusPass(w);
			lang.onsetPass(w);
			for (ISyllable s : w.getSyllables()) {
				String p=s.toString();
				if (!pinyinMap.containsKey(p)) {
					pinyinMap.put(p, new PinYin(p));
				} else {
					pinyinMap.get(p).icreaseFreq();
				}
			}
									
		}
	}

	/**
	 * 对文本进行预分类
	 * 
	 * @param files
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	private void readVocabWithSupervised(File[] files) throws IOException {
		for (int category = 0; category < files.length; category++) {
			// 对多个文件学习
			MapCount<String> mc = new MapCount<>();
			try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(files[category])))) {
				String temp = null;
				while ((temp = br.readLine()) != null) {
					String[] split = temp.split(" ");
					trainWordsCount += split.length;
					for (String string : split) {
						mc.add(string);
					}
				}
			}
			for (Entry<String, Integer> element : mc.get().entrySet()) {
				double tarFreq = (double) element.getValue() / mc.size();
				if (wordMap.get(element.getKey()) != null) {
					double srcFreq = wordMap.get(element.getKey()).freq;
					if (srcFreq >= tarFreq) {
						continue;
					} else {
						Neuron wordNeuron = wordMap.get(element.getKey());
						wordNeuron.category = category;
						wordNeuron.freq = tarFreq;
					}
				} else {
					wordMap.put(element.getKey(), new WordNeuron(element.getKey(), tarFreq, category, layerSize));
				}
			}
		}
	}

	/**
	 * Precompute the exp() table f(x) = x / (x + 1)
	 */
	private void createExpTable() {
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
	}

	/**
	 * 构建Negative Sampling用的负采样表
	 */
	private void initUnigramTable() {
		int i;
		long train_words_pow = 0;
		double dl = 0, power = 0.75;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			train_words_pow += Math.pow(word.getValue().freq, power);
		}

		i = 0;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			if (i == 0) {
				dl = Math.pow(word.getValue().freq, power) / train_words_pow;
			} else {
				dl += Math.pow(word.getValue().freq, power) / train_words_pow;
			}
			while (i / (double) table_size < dl && i < table_size) {
				table[i++] = word.getKey();
			}
		}
	}

	/**
	 * 根据文件学习
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void learnFile(File file) throws IOException {
		readVocab(file);
		ReduceVocab();
		getPinOrChar();

		if (negative > 0) {
			initUnigramTable();
		}
		trainModel(file);
	}

	/**
	 * 根据预分类的文件学习
	 * 
	 * @param summaryFile
	 *            合并文件
	 * @param classifiedFiles
	 *            分类文件
	 * @throws IOException
	 */
	public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException {
		readVocabWithSupervised(classifiedFiles);
		new Haffman(layerSize).make(wordMap.values());
		// 查找每个神经元
		for (Neuron neuron : wordMap.values()) {
			((WordNeuron) neuron).makeNeurons();
		}
		trainModel(summaryFile);
	}

	/**
	 * 保存模型
	 */
	public void saveModel(File file) {
		// TODO Auto-generated method stub

		try {
			OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(file), "utf-8");
			BufferedWriter write = new BufferedWriter(out);

			write.write(wa + " " + wb + "\r\n");
			write.flush();
			// write the word vector to file
			write.write("the word vector is \r\n");//输入向量作为词向量
			write.flush();

			double[] syn0 = new double[layerSize];
			for (Entry<String, Neuron> element : wordMap.entrySet()) {
				write.write(element.getKey() + " ");
				write.flush();

				// syn0 = ((WordNeuron) element.getValue()).syn0;
				for (int i = 0; i < layerSize; i++) {
					syn0[i] = ((WordNeuron) element.getValue()).syn0[i];
				}

				for (int j = 0; j < layerSize; j++) {
					write.write(syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();
			}

			write.write("the temp vector is \r\n");//输出向量作为词向量
			write.flush();

			for (Entry<String, Neuron> element : wordMap.entrySet()) {
				write.write(element.getKey() + " ");
				write.flush();

				// syn0 = ((WordNeuron) element.getValue()).syn0;
				for (int i = 0; i < layerSize; i++) {
					syn0[i] = ((WordNeuron) element.getValue()).tempsyn0[i];
				}

				for (int j = 0; j < layerSize; j++) {
					write.write(syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();
			}

			// write the pinyin vector to the file
			write.write("the pinyin vector is \r\n");
			write.flush();

			for (Entry<String, PinYin> py : pinyinMap.entrySet()) {
				write.write(py.getValue().py + " ");
				write.flush();
				for (int j = 0; j < layerSize; j++) {
					write.write(py.getValue().syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();

			}

			wordMap.clear();
			pinyinMap.clear();
			write.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public int getLayerSize() {
		return layerSize;
	}

	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}

	public int getWindow() {
		return window;
	}

	public void setWindow(int window) {
		this.window = window;
	}

	public double getSample() {
		return sample;
	}

	public void setSample(double sample) {
		this.sample = sample;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
		this.startingAlpha = alpha;
	}

	public static int getParam(String para, String[] args) {
		int i;
		for (i = 0; i < args.length; i++) {
			if (args[i].equals(para)) {
				return i + 1;
			}
		}
		return -1;
	}

	public static void main(String[] args) throws IOException {

		int count = 10, k;
		String inputBaseFile = null;
		String outputBaseFile = null;
		if ((k = getParam("-count", args)) > 0) {
			count = Integer.valueOf(args[k]);
		}
		System.out.println("the count of the train is " + count);

		if ((k = getParam("-input", args)) > 0) {
			inputBaseFile = args[k];
		}
		if ((k = getParam("-output", args)) > 0) {
			outputBaseFile = args[k];
		}
		System.out.println("input base file: " + inputBaseFile);
		System.out.println("output base file: " + outputBaseFile);

		for (int i = 1; i <= count; i++) {
			Serial learn = new Serial();
			int j;
			if ((j = getParam("-negative", args)) > 0) {
				learn.negative = Integer.valueOf(args[j]);
			}
			System.out.println("negative: " + learn.negative);

			long start = System.currentTimeMillis();
			learn.learnFile(new File(inputBaseFile + "pure"));
			System.out.println("use time " + (System.currentTimeMillis() - start));

			learn.saveModel(new File(outputBaseFile + "0.1" + "pycbow.txt"));
		}
	}
}
