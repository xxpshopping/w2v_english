package analogy;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import Test.WeightedWords;
import syllabify.model.English;
import syllabify.model.ISyllable;
import syllabify.model.Word;

public class Analogy {
	
	Map<String, Integer> words = new HashMap<>();
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> twords = new HashMap<>(); // �ʵ�
	ArrayList<ArrayList<Double>> tC = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> pinyins = new HashMap<>(); // pinyin map
	ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();
	int dim = 100;
	String trainfile;
	int range;
	
	public Analogy(String trainfile, int range) {
		this.trainfile = trainfile;
		this.range = range;
	}
	
	public double accuracyAnalogy() {
		readVec();
		InputStreamReader input;
		double total = 0;
		double right = 0;
		double[] a = new double[dim];
		double[] a1 = new double[dim];
		double[] b = new double[dim];
		ArrayList<WeightedWords<String>> top5;
		try {
			input = new InputStreamReader(new FileInputStream("questions-words.txt"), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			while((line = read.readLine()) != null) {
				if(line.contains(":"))
					continue;
//				if(line.equals(": capital-common-countries") || line.equals(": family") || line.equals(": city-in-state")) {
//					continue;
//				}
				String[] split = line.split(" ");
				Integer w1 = twords.get(split[0]);
				Integer w2 = twords.get(split[1]);
				Integer w3 = twords.get(split[2]);
				Integer w4 = twords.get(split[3]);
				if(w1 != null && w2 != null && w3 !=null && w4 != null) {
					double[] temp = new double[100];
					
					for (int i = 0; i < dim; i++) {
						a[i] = tC.get(w1).get(i);
						a1[i] = tC.get(w2).get(i);
						b[i] = tC.get(w3).get(i);
					}
					top5 = findSimiliar(a, a1, b);
					total++;
					for(int i = 0; i < top5.size(); i++) {
						System.out.println(top5.get(i).num);
						if(top5.get(i).num.equals(split[3])) {
							right++;
							break;
						}
					}
					System.out.println("-----------------------");
//					System.out.println(split[0] + " " + split[1] + " " + split[2] + " " + top5.get(0).num + " " + top5.get(1).num + " " + top5.get(2).num + " " + top5.get(3).num + " " + top5.get(4).num);
					
				}
				
			}
		} catch (UnsupportedEncodingException | FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("right:" + right + "total:" + total);
		return right / total * 100;
	}
	
	//get the analogy result
	ArrayList<WeightedWords<String>> findSimiliar(double[] a, double[] a1, double[] b) {
		double sa, sa1, sb;
		ArrayList<WeightedWords<String>> top5 = new ArrayList<WeightedWords<String>>();
		double thetaa = 0.0, thetaa1 = 0.0, thetab=0.0, thetaw;
		
		for (int j = 0; j < dim; j++) {
			thetaa = thetaa + a[j] * a[j];
			thetaa1 = thetaa1 + a1[j] * a1[j];
			thetab = thetab + b[j] * b[j];
		}
		
		for (Entry<String, Integer> e: twords.entrySet()) {
			sa = 0.0;
			sa1 = 0.0;
			sb = 0.0;
			thetaw = 0.0;
			
			for (int j2 = 0; j2 < dim; j2++) {
				double right = tC.get(e.getValue()).get(j2);
				sa = sa + a[j2] * right;
				thetaw = thetaw + right * right;
				
				sa1 = sa1 + a1[j2] * right;
				
				sb = sb + b[j2] * right;
			}
			sa = sa / (Math.sqrt(thetaw) * Math.sqrt(thetaa));
			sa1 = sa1 / (Math.sqrt(thetaw) * Math.sqrt(thetaa1));
			sb = sb / (Math.sqrt(thetaw) * Math.sqrt(thetab));
			
			double s = sb + sa1 - sa;
			
			top5.add(new WeightedWords<String>(e.getKey(), s));
			Collections.sort(top5);
			if(top5.size() > range) {
				top5.remove(range);
			}
		}
		return top5;
	}
	
	public void readVec() {
		// ��ȡ��ѵ���õĴ�����
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0, pnum = 0, tnum = 0;
			line = read.readLine();
			factors = line.split(" ");

			while ((line = read.readLine()) != null) {
				if (line.equals("the word vector is ")) {
					continue;
				}
				if (line.equals("the pinyin vector is ") || line.equals("the temp vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.put(factors[0], num);//words存放着单词，和单词索引
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);//C存放单词向量
				num++;
			}
			
			while ((line = read.readLine()) != null) {
				if (line.equals("the pinyin vector is ")) {
					break;
				}
				factors = line.split(" ");
				twords.put(factors[0], tnum);//twords放单词和索引
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				tC.add(vec);//放单词辅助向量
				tnum++;
			}

			while ((line = read.readLine()) != null) {
				factors = line.split(" ");
				pinyins.put(factors[0], pnum);//pinyins放音
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				P.add(vec);//P放音向量
				pnum++;
			}
			
			int charsize;
			for (Entry<String, Integer> e : twords.entrySet()) {
				//List<Pinyin> pya = HanLP.convertToPinyinList(e.getKey());
				charsize = e.getKey().length();
				double[] temp = new double[100];
				English lang = new English();
				Word w = new Word(e.getKey());
				lang.nucleusPass(w);
			    lang.onsetPass(w);
			    for (ISyllable s : w.getSyllables()) {
			    	int indexp = pinyins.get(s.toString());
			    	for (int i = 0; i < dim; i++) {
			    		temp[i] += P.get(indexp).get(i)/charsize;
			    		}
			        }
			    
				for (int i = 0; i < dim; i++) {
					temp[i] = tC.get(e.getValue()).get(i) - 0.5 * temp[i];
					tC.get(e.getValue()).set(i, temp[i]);
				}
			}
			
			System.out.println("C size " + C.size() + " " + num);
			System.out.println("tC size " + tC.size() + " " + tnum);
			System.out.println("P size " + P.size() + " " + pnum);
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
}
