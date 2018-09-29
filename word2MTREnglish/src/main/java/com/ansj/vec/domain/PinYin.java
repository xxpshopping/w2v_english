package com.ansj.vec.domain;

import java.util.Random;

public class PinYin extends Neuron {
	public String py; //具体拼音的值
	final int dim = 100;
	public Double[] syn0 = new Double[dim]; // 音向量的值
	public Double[] tempsyn0 = new Double[dim]; //负采样的辅助参数
	
	public PinYin(String p) {
		this.py = p;
		Random random = new Random();
		for (int j = 0; j < dim; j++) {
			syn0[j] = (random.nextDouble() - 0.5) / dim;
			tempsyn0[j] = (random.nextDouble() - 0.5) / dim;
		}
		this.freq = 0; 
	}
	
	public void icreaseFreq() {
		this.freq++;
	}
}
