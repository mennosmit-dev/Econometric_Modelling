import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;

import java.util.List;

/**
 * A utility class for sampling data from a probability distribution.
 */
public class DistributionSampler {
    
    private final RealDistribution distribution;

    /**
     * Constructs a sampler for a specified distribution.
     *
     * @param distribution the distribution to sample from
     */
    public DistributionSampler(RealDistribution distribution) {
        this.distribution = distribution;
    }

    /**
     * Generates a table of sampled data using the provided distribution.
     *
     * @param seed         seed for the random number generator
     * @param rowCount     number of rows of data to sample
     * @param columnNames  list of column names
     * @return a TableData object containing the sampled data
     */
    public TableData<Double> sampleData(long seed, int rowCount, List<String> columnNames) {
        distribution.reseedRandomGenerator(seed);
        int colCount = columnNames.size();
        double[][] sampledData = new double[rowCount][colCount];

        for (int j = 0; j < colCount; j++) {
            for (int i = 0; i < rowCount; i++) {
                sampledData[i][j] = distribution.sample();
            }
        }

        return new NumericTable(columnNames, sampledData);
    }

    /**
     * Creates a DistributionSampler with a uniform distribution.
     *
     * @param lowerBound the lower bound
     * @param upperBound the upper bound
     * @return a DistributionSampler for the uniform distribution
     */
    public static DistributionSampler uniform(double lowerBound, double upperBound) {
        return new DistributionSampler(new UniformRealDistribution(lowerBound, upperBound));
    }

    /**
     * Creates a DistributionSampler with a Gaussian (normal) distribution.
     *
     * @param mean               the mean of the distribution
     * @param standardDeviation  the standard deviation of the distribution
     * @return a DistributionSampler for the normal distribution
     */
    public static DistributionSampler gaussian(double mean, double standardDeviation) {
        return new DistributionSampler(new NormalDistribution(mean, standardDeviation));
    }

    /**
     * Creates a DistributionSampler with an exponential distribution.
     *
     * @param mean the mean of the distribution
     * @return a DistributionSampler for the exponential distribution
     */
    public static DistributionSampler exponential(double mean) {
        return new DistributionSampler(new ExponentialDistribution(mean));
    }
}
