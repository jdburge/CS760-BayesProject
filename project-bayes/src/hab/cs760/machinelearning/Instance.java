package hab.cs760.machinelearning;

import java.util.List;

/**
 * Created by hannah on 9/29/17.
 */
public class Instance {
	private final FeatureVector featureVector;
	public String actualLabel;
	private NominalFeature classLabelFeature;

	public Instance() {
		featureVector = new FeatureVector();
	}

	public static Instance makeInstance(String line, List<Feature> featureList) {
		Instance instance = new Instance();

		String[] values = line.split(",");

		for (int i = 0; i < values.length; i++) {
			NominalFeature feature = (NominalFeature) featureList.get(i);
			String featureValue = values[i].replace("\'", "");
			if (featureValue.endsWith(".")) {
				featureValue = featureValue.substring(0, featureValue.length() - 1);
			}
			featureValue = featureValue.trim();
			instance.addFeature(feature, featureValue);
		}

		return instance;
	}

	public void setClassLabel(int labelValueIndex) {
		featureVector.setValueForFeature(classLabelFeature, classLabelFeature.possibleValues.get(labelValueIndex));
	}

	public String getClassLabel() {
		return featureVector.get(classLabelFeature);
	}

	public void revertClassLabel() {
		featureVector.setValueForFeature(classLabelFeature, actualLabel);

	}

	public void addFeature(NominalFeature feature, String value) {
		featureVector.add(feature, value);
		if (feature.index == featureVector.classLabelIndex()) {
			classLabelFeature = feature;
			actualLabel = value;
		}
	}

	public String getFeatureValue(NominalFeature feature) {
		return featureVector.get(feature);
	}


}
