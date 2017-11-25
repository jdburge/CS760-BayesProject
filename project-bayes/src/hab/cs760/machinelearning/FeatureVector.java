package hab.cs760.machinelearning;

import java.util.ArrayList;

/**
 * Created by hannah on 9/29/17.
 */
class FeatureVector {
	private final ArrayList<Object> features;

	public FeatureVector() {
		features = new ArrayList<>();
	}

	public double get(NumericFeature feature) {
		return (Double) features.get(feature.index);
	}

	public String get(NominalFeature feature) {
		return (String) features.get(feature.index);
	}

	public void add(NumericFeature feature, Double value) {
		features.add(feature.index, value);
	}

	public void add(NominalFeature feature, String value) {
		features.add(feature.index, value);
	}
	public void setValueForFeature(NominalFeature feature, String value) {
		if (features.get(feature.index) != null) features.remove(feature.index);
		add(feature, value);
	}

	public int classLabelIndex() {
		return features.size() - 1;
	}
}
