package Test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.recognition.impl.StopRecognition;
import org.ansj.splitWord.analysis.ToAnalysis;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class TextClassification {
	
//	private int small = 294;
//	private int large = 6121;
	Feature[][] featureMatrix;
	Feature[][] testfeatureMatrix;
	double[] trainTarget;
	double[] testTarget;
	ArrayList<Feature[]> trainFeatureList = new ArrayList<Feature[]>();
	ArrayList<Feature[]> testFeatureList = new ArrayList<Feature[]>();
	ArrayList<Double> trainTargetList = new ArrayList<Double>();
	ArrayList<Double> testTargetList = new ArrayList<Double>();
	int trainNumber;
	int testNumber;
	//���Լ���û�������
	double[] typeSize;
	//Ԥ��õ���ÿ������
	double[] preSize;
	File[] categories;
	
	int dim = 100;
	Map<String, Integer> words = new HashMap<>();
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> twords = new HashMap<>(); // �ʵ�
	ArrayList<ArrayList<Double>> tC = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> pinyins = new HashMap<>(); // pinyin map
	ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();
	String trainfile;
	
	public TextClassification(String trainfile) {
		this.trainfile = trainfile;
	}

	public void getDocVec(String textPath, String encode) {
		try {
			File dir = new File(textPath);
			categories = dir.listFiles();
			Arrays.sort(categories);
			int countOfCategory = categories.length;
			preSize = new double[countOfCategory];
			typeSize = new double[countOfCategory];
			for (int k = 0; k < countOfCategory; k++) {
				File[] files = categories[k].listFiles();
				Arrays.sort(files);
				int countOfFile = files.length;
				int countOfTestType = 0;
				for (int i = 0; i < countOfFile; i++) {

					// get the text to the String
					InputStreamReader input = new InputStreamReader(new FileInputStream(files[i]), encode);
					BufferedReader read = new BufferedReader(input);
					String line;
					String text = "";
					while ((line = read.readLine()) != null) {
						text = text + line;
					}
					// split the text
					StopRecognition stop = new StopRecognition();
					stop.insertStopNatures("w");
					Result result = ToAnalysis.parse(text).recognition(stop);

					// get the average Vector as document vector
					int countOfWords = 0;
					double[] docVec = new double[dim];
					for (Term term : result) {
						String item = term.getName().trim();
						if (item.length() > 0 && item.matches("[\u4e00-\u9fa5]+")) {
							Integer index = words.get(item);							
							if (index != null) {
								
								
//								double charsizea = 0;
//								List<Pinyin> pya = HanLP.convertToPinyinList(item);
//								charsizea = item.length();
//					
//								for (Pinyin pinyin : pya) {
//									if (pinyin == null) {
//										System.out.println("worda pinyin not exit");
//										continue;
//									}
//					
//									int indexp = pinyins.get(pinyin.toString());
//									for (int j = 0; j < dim; j++) {
//										docVec[j] = docVec[j] + P.get(indexp).get(j)/charsizea;
//									}
//								}
								
								
								countOfWords++;
								for (int j = 0; j < dim; j++) {
									docVec[j] += C.get(index).get(j);
								}
							}
						}
					}
					
					if (countOfWords != 0) {
						Feature[] feature = new Feature[dim];
						for (int j = 0; j < dim; j++) {
							docVec[j] /= countOfWords;
							feature[j] = new FeatureNode(j + 1, docVec[j]);
						}
						if (i < countOfFile * 0.8) {
							trainFeatureList.add(feature);
							trainTargetList.add((double) k);
						} else {
							countOfTestType++;
							testFeatureList.add(feature);
							testTargetList.add((double) k);
						}
					}
				}
				typeSize[k] = countOfTestType;
			}
			
			trainNumber = trainFeatureList.size();
			testNumber = testFeatureList.size();
			featureMatrix = new Feature[trainNumber][];
			testfeatureMatrix = new Feature[testNumber][];
			trainTarget = new double[trainNumber];
			testTarget = new double[testNumber];
			
			for (int i = 0; i < trainNumber; i++) {
				featureMatrix[i] = trainFeatureList.get(i);
				trainTarget[i] = trainTargetList.get(i);
			}
			
			for (int i = 0; i < testNumber; i++) {
				testfeatureMatrix[i] = testFeatureList.get(i);
				testTarget[i] = testTargetList.get(i);
			}
			testFeatureList.clear();
			testTargetList.clear();
			trainFeatureList.clear();
			trainTargetList.clear();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
//					wa = Double.parseDouble(factors[0]);
//					wb = Double.parseDouble(factors[1]);
					while ((line = read.readLine()) != null) {
						if (line.equals("the word vector is ")) {
							continue;
						}
						if (line.equals("the pinyin vector is ") || line.equals("the temp vector is ")) {
							break;
						}
						factors = line.split(" ");
						words.put(factors[0], num);
						ArrayList<Double> vec = new ArrayList<Double>();
						for (int i = 1; i <= dim; i++) {
							vec.add(Double.valueOf(factors[i]));
						}
						C.add(vec);
						num++;
					}
					
					while ((line = read.readLine()) != null) {
						if (line.equals("the pinyin vector is ")) {
							break;
						}
						factors = line.split(" ");
						twords.put(factors[0], tnum);
						ArrayList<Double> vec = new ArrayList<Double>();
						for (int i = 1; i <= dim; i++) {
							vec.add(Double.valueOf(factors[i]));
						}
						tC.add(vec);
						tnum++;
					}

					while ((line = read.readLine()) != null) {
						factors = line.split(" ");
						pinyins.put(factors[0], pnum);
						ArrayList<Double> vec = new ArrayList<Double>();
						for (int i = 1; i <= dim; i++) {
							vec.add(Double.valueOf(factors[i]));
						}
						P.add(vec);
						pnum++;
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

	public double calculateAcc(String textPath, String encode, BufferedWriter write) throws IOException {
		// loading train data
		readVec();
		getDocVec(textPath, encode);

		Problem problem = new Problem();
		problem.l = trainNumber; // number of training examples��ѵ��������
		problem.n = dim; // number of features������ά��
		problem.x = featureMatrix; // feature nodes����������
		problem.y = trainTarget; // target values�����

		SolverType solver = SolverType.L2R_L2LOSS_SVC_DUAL; // -s 1
		double C = 1.0; // cost of constraints violation
		double eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);
		File modelFile = new File("model");
		int i = 0;
		int j = 0;
		
		double[] accurateType = new double[typeSize.length];
		try {
			model.save(modelFile);
			// load model or use it directly
			model = Model.load(modelFile);

			for (j = 0; j < testfeatureMatrix.length; j++) {
				double prediction = Linear.predict(model, testfeatureMatrix[j]);
				preSize[(int) prediction]++;
				if(prediction == testTarget[j]) {
					accurateType[(int) prediction]++;
					i++;
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double[] acc = new double[typeSize.length];
		double[] recall = new double[typeSize.length];
		double[] f = new double[typeSize.length];
		for (int typeIndex = 0; typeIndex < typeSize.length; typeIndex++) {
			acc[typeIndex] = accurateType[typeIndex] / preSize[typeIndex] * 100;
			recall[typeIndex] = accurateType[typeIndex] / typeSize[typeIndex] * 100;
			f[typeIndex] = 2 * acc[typeIndex] * recall[typeIndex] / (acc[typeIndex] + recall[typeIndex]);
			write.write(categories[typeIndex].getName() + " " + acc[typeIndex] + " " + recall[typeIndex] + " " + f[typeIndex] + " ");
			write.flush();
		}
		return (double) i / (double) testNumber * 100;
	}
}

