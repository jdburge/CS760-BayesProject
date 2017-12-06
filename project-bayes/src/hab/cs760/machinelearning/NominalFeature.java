package hab.cs760.machinelearning;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Created by hannah on 9/29/17.
 */
public class NominalFeature extends Feature {
	public final List<String> possibleValues;

	public NominalFeature(String name, int index, String possibleValueString) {
		super(name, index);

		possibleValueString = possibleValueString.replace("{", "");
		possibleValueString = possibleValueString.replace("}", "");
		possibleValueString = possibleValueString.trim();
		possibleValueString = possibleValueString.replace("\'", "");
		possibleValues = Arrays.asList(possibleValueString.split(","));
		for (int i = 0; i < possibleValues.size(); i++) {
			possibleValues.set(i, possibleValues.get(i).trim());
		}
	}

	public NominalFeature(String name, int index, String... possibleValues) {
		super(name, index);
		this.possibleValues = Arrays.asList(possibleValues);
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		NominalFeature that = (NominalFeature) o;

		if (!Objects.equals(possibleValues, that.possibleValues)) return false;

		return super.equals(o);
	}

	@Override
	public int hashCode() {
		return possibleValues != null ? possibleValues.hashCode() : 0;
	}

	@Override
	public String toString() {
		return super.toString() + String.format(", Possible values: %s", possibleValues);
	}

}
