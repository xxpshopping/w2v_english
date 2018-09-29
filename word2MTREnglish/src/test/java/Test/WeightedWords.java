package Test;

public class WeightedWords<T> implements Comparable<WeightedWords<T>> {
	
	public T num;
	public double weight;
	
	public WeightedWords(T n, double w) {
		this.num = n;
		this.weight = w;
	}

	public int compareTo(WeightedWords<T> arg0) {
		// TODO Auto-generated method stub
		if(arg0.weight > this.weight) {
			return 1;
		}
		
		if(arg0.weight < this.weight) {
			return -1;
		}
		return 0;
	}
}
