package com.ansj.vec;


import java.io.BufferedReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.sql.Time;
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
public class count2 {

	private Map<String, Neuron> wordMap = new HashMap<String, Neuron>(); 
	private Map<String, PinYin> pinyinMap = new HashMap<String, PinYin>();
	private Map<String, String> transMap = new HashMap<String, String>();
	private Map<String, Integer> words = new HashMap<String, Integer>();
	private ArrayList<ArrayList<Double>> V = new ArrayList<ArrayList<Double>>();
	
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

	public count2(Boolean isCbow, Boolean ispinyin, Integer layerSize, Integer window, Double alpha, Double sample) {
		
		if (layerSize != null)
			this.layerSize = layerSize;
		if (window != null)
			this.window = window;
		if (alpha != null)
			this.alpha = alpha;
		if (sample != null)
			this.sample = sample;
	}
	public count2() {
		
	}
	
	/**
	 * 将音标文件存储到map<string,int>
	 * int指向arraylist<arraylist<double>>
	 */
	private void readVec(File file) {
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(file), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0;
			line = read.readLine();
			factors = line.split(" ");
			// wa = Double.parseDouble(factors[0]);
			// wb = Double.parseDouble(factors[1]);
			while ((line = read.readLine()) != null) {
				factors = line.split(" ");
				words.put(factors[0], num);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= layerSize; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				V.add(vec);
				num++;
			}
	    }
		catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * trainModel
	 * 
	 * @throws IOException
	 */
	private void getTrans(File pronunciation) {
		try {
			BufferedReader read = new BufferedReader(
					new InputStreamReader(new FileInputStream(pronunciation), "utf-8"));
			String line;
			String[] words;
			while ((line = read.readLine()) != null) {
				words = line.split(" ");
				transMap.put(words[0].trim(), words[1].trim());
				
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	private void Computer(File filein,File fileout,Double a) throws IOException {
		String pronun;
		int num=0;
		int i=0;
		
		OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(fileout), "utf-8");
		BufferedWriter write = new BufferedWriter(out);
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filein)));
		String temp = null;
		long starttime1 = System.currentTimeMillis();
		while ((temp = br.readLine()) != null) {	
			i++;
			String[] split = temp.split(" ");
			pronun = transMap.get(split[0]);
			if(words.get(pronun)==null)
				continue;
			num=words.get(pronun);
			write.write(split[0]+ " ");
			for (int j = 1; j <=layerSize; j++) {
				BigDecimal b1 = new BigDecimal(split[j]);
				BigDecimal b2 = new BigDecimal(V.get(num).get(j-1));
				BigDecimal alphy = new BigDecimal(a);
				BigDecimal beta = new BigDecimal(1-a);
				b1=b1.multiply(alphy);
				b2=b2.multiply(beta);
				b1 = b1.add(b2);
				b1 = b1.setScale(5,RoundingMode.HALF_DOWN);
				write.write(b1 + " ");
				write.flush();
			}
			write.write("\n");
			write.flush();
			if(i%10000==0)
				System.out.println("第"+i+"行  "+split[0]);
			
			
		}
		long starttime4 = System.currentTimeMillis();
		br.close();
		write.close();
		System.out.println("总耗时："+(starttime4-starttime1)/1000+"秒");
	}

	public void learnFile(File filein,File fileout,Double a) throws IOException {
		getTrans(new File("E:\\文本表示语料\\English\\part\\partDic"));
		System.out.println("getTrans is over");
		readVec(new File("E:\\文本表示语料\\English\\part\\音标n-gram\\transfervec.vec"));
		System.out.println("音标文件已存入map");
		Computer(filein, fileout, a);

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
		File filein=new File("E:\\文本表示语料\\English\\part\\fasttextmodel.vec");
		File fileout=new File("E:\\文本表示语料\\English\\part\\音标n-gram\\0.01count.txt");
		
		//Double a=0.6;
		count2 learn = new count2();
		learn.learnFile(filein,fileout,0.99);
		System.out.print("transfer is over");
		
	}
}

