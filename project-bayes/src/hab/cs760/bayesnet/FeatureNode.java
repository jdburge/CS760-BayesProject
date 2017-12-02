package hab.cs760.bayesnet;

import com.sun.istack.internal.Nullable;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hannah on 10/30/17.
 */
public class FeatureNode extends Node {
	public Edge labelNodeEdge;
	private final Map<String, Map<String, Map<String, Double>>> probabilities;

	FeatureNode(NominalFeature feature) {
		super(feature);
		this.labelNodeEdge = null;
		probabilities = new HashMap<>();
	}

	@Override
	protected void calculateProbabilitiesForThisNode(List<Instance> instances) {
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;

		Node treeParent = getTreeParent();
		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			probabilities.put(labelFeatureValue, block);

			if (treeParent != null) {
				NominalFeature treeParentFeature = treeParent.feature;

				for (String treeParentFeatureValue : treeParentFeature.possibleValues) {
					Map<String, Double> row = rowWithCriteria(instances, new InstanceCounter.Criterion(labelFeature, labelFeatureValue), new InstanceCounter.Criterion(treeParentFeature, treeParentFeatureValue));
					block.put(treeParentFeatureValue, row);
				}
			} else {
				Map<String, Double> row = rowWithCriteria(instances, new InstanceCounter.Criterion
						(labelFeature, labelFeatureValue));
				block.put("", row);
			}
		}
	}

	private Map<String, Double> rowWithCriteria(List<Instance> instances, InstanceCounter.Criterion... criteria) {
		Map<String, Double> row = new HashMap<>();

		for (String featureValue : feature.possibleValues) {
			double probability = InstanceCounter.probabilityOfCriterionGivenCriteria(feature,
					featureValue, instances, criteria);
			row.put(featureValue, probability);
		}
		return row;
	}

	@Nullable
	private Node getTreeParent() {
		for (Edge edge : connectedEdges) {
			if (edge.end() == this) {
				return edge.start();
			}
		}
		return null;
	}

	@Override
	public double probabilityOf(String labelNodeValue, String treeParentValue, String
			thisFeatureValue) {
		if (getTreeParent() != null) {
			assert getTreeParent().feature.possibleValues.contains(treeParentValue);
			return probabilities.get(labelNodeValue).get(treeParentValue).get(thisFeatureValue);
		}
		return probabilities.get(labelNodeValue).get("").get(thisFeatureValue);
	}
	
	public void makeFair() {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap();
		FeatureNode sensativeNode = (FeatureNode)getTreeParent();
		NominalFeature sensativeFeature = sensativeNode.feature;
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb = -1.0;
		double probSensativeFeature = -1.0;
		double newProb = -1.0;
		double weightedAverage = -1.0;
		
		for(String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);
			
			for(String sensativeFeatureValue : sensativeFeature.possibleValues) {
				Map<String, Double> row = new HashMap();
				probSensativeFeature = getProbOfSensativeFeature(sensativeNode, sensativeFeatureValue);
				weightedAverage = getWeightedAverage(labelFeatureValue,sensativeFeatureValue,sensativeFeature);
				
				for(String thisFeatureValue : this.feature.possibleValues) {
					baseProb = probabilityOf(labelFeatureValue,sensativeFeatureValue,thisFeatureValue);
					newProb = (baseProb*probSensativeFeature) + ((1-probSensativeFeature)*weightedAverage);
					row.put(thisFeatureValue, newProb);
				}
				
				
				
				block.put(sensativeFeatureValue, row);
			}
			
		}
		
		probabilities = newProbabilities; 
	}
	
	private double getProbOfSensativeFeature(FeatureNode sensativeNode, String choice) {
		double runningProb =0.0;
		NominalFeature labelFeature = labelNodeEdge.start().feature;
		for(String labelFeatureValue : labelFeature.possibleValues) {
			runningProb += sensativeNode.probabilityOf(labelFeatureValue,"",choice);
		}
		return runningProb;
	}
	
	private double getWeightedAverage(String labelFeatureValue, String sensativeFeatureValue, NominalFeature sensativeFeature) {
		double runningProb = 0.0;
		for(String otherSensFeatureVal : sensativeFeature.possibleValues) {
			if(!otherSensFeatureVal.equals(sensativeFeatureValue)) {
				for(String thisFeatureValue : this.feature.possibleValues) {
					runningProb += probabilityOf(labelFeatureValue,sensativeFeatureValue, thisFeatureValue);
				}
			}
		}
		return runningProb/(sensativeFeature.possibleValues.size()*this.feature.possibleValues.size());
	}
}
