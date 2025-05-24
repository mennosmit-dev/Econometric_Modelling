import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TTest;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A utility class for performing basic statistical analyses on a numeric dataset.
 */
public class DataAnalytics implements TableAnalytics {

    private final NumericTable dataset;

    public DataAnalytics(NumericTable dataset) {
        this.dataset = dataset;
    }

    /**
     * Converts a column of the dataset into a double array.
     *
     * @param columnName the name of the column
     * @return an array of doubles representing the column's values
     */
    public double[] columnToArray(String columnName) {
        List<Double> columnData = dataset.getColumnVector(columnName).getValues();
        double[] arrayData = new double[columnData.size()];
        for (int i = 0; i < columnData.size(); i++) {
            arrayData[i] = columnData.get(i);
        }
        return arrayData;
    }

    @Override
    public double tTest(String column, double mean) {
        TTest tTest = new TTest();
        return tTest.tTest(mean, columnToArray(column));
    }

    @Override
    public double tTest(String column1, String column2) {
        TTest tTest = new TTest();
        return tTest.tTest(columnToArray(column1), columnToArray(column2));
    }

    @Override
    public double pearsonsCorrelation(String column1, String column2) {
        PearsonsCorrelation correlation = new PearsonsCorrelation();
        return correlation.correlation(columnToArray(column1), columnToArray(column2));
    }

    @Override
    public DescriptiveStatistics describe(String column) {
        return new DescriptiveStatistics(columnToArray(column));
    }

    @Override
    public Map<String, Double> estimateLinearModel(String dependent, List<String> independents) {
        int rowCount = dataset.rowCount();
        int numFeatures = independents.size();
        double[][] independentData = new double[rowCount][numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double[] columnData = columnToArray(independents.get(j));
            for (int i = 0; i < rowCount; i++) {
                independentData[i][j] = columnData[i];
            }
        }

        OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
        regression.newSampleData(columnToArray(dependent), independentData);

        double[] parameters = regression.estimateRegressionParameters();
        Map<String, Double> modelParameters = new LinkedHashMap<>();
        modelParameters.put("intercept", parameters[0]);

        for (int i = 1; i < parameters.length; i++) {
            modelParameters.put(independents.get(i - 1), parameters[i]);
        }

        return modelParameters;
    }
}
