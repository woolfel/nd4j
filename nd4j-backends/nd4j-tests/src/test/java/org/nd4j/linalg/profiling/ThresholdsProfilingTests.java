package org.nd4j.linalg.profiling;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Ignore
@RunWith(Parameterized.class)
public class ThresholdsProfilingTests  extends BaseNd4jTest {
    protected int[] thresholds = new int[]{1, 1024, 2048, 8192, 16384, 65536, 131072, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432};
    protected List<int[]> shapes = Arrays.asList(new int[]{100, 100}, new int[]{1000, 1000}, new int[]{2500, 2500}, new int[]{5000, 5000}); //, new int[]{10000, 10000});
    protected int NUM_ARRAYS = 1;
    protected int NUM_TRIES = 500;
    protected int NUM_WARM = 500;

    DataBuffer.Type initialType;

    public ThresholdsProfilingTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);


        // warmup
        log.info("*********************** WARM-UP PHASE");
        INDArray warm = Nd4j.create(5000,5000);
        for (int i = 0; i < NUM_WARM; i++)
            warm.addi(i);
    }

    @After
    public void shutUp() {
        Nd4j.setDataType(initialType);
    }


    @Test
    public void testScalarThresholds1() throws Exception {
        // this test is irrelevant on CUDA :(
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        log.info("*********************** SCALAR {}", Nd4j.dataType());

        for (int[] shape: shapes) {
            List<INDArray> X = new ArrayList<>();
            for (int i = 0; i < NUM_ARRAYS; i++) {
                X.add(Nd4j.rand(shape));
            }

            for (int threshold: thresholds) {
                List<Long> times = new ArrayList<>();
                long avg = 0;

                NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(threshold);

                for (int t = 0; t < NUM_TRIES; t++) {
                    int opIdx = t % NUM_ARRAYS;
                    INDArray x = X.get(opIdx).dup();

                    long time1 = System.nanoTime();
                    x.muli(1.0f);
                    long time2 = System.nanoTime();
                    times.add(time2 - time1);
                    avg += (time2 - time1);
                }

                avg /= NUM_TRIES;
                Collections.sort(times);

                log.info("Shape: {}; Threshold: {}; Average: {} us; p50: {} us", Arrays.toString(shape), threshold, avg / 1000, times.get(times.size() / 2) / 1000);
            }
        }
    }

    @Test
    public void testPairwiseThresholds1() throws Exception {
        // this test is irrelevant on CUDA :(
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        log.info("*********************** PAIRWISE {}", Nd4j.dataType());

        for (int[] shape: shapes) {
            List<INDArray> X = new ArrayList<>();
            List<INDArray> Y = new ArrayList<>();
            for (int i = 0; i < NUM_ARRAYS; i++) {
                X.add(Nd4j.rand(shape));
                Y.add(Nd4j.rand(shape));
            }

            for (int threshold: thresholds) {
                List<Long> times = new ArrayList<>();
                long avg = 0;

                NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(threshold);

                for (int t = 0; t < NUM_TRIES; t++) {
                    int opIdx = t % NUM_ARRAYS;
                    INDArray x = X.get(opIdx).dup();
                    INDArray y = X.get(opIdx).dup();

                    long time1 = System.nanoTime();
                    x.addi(y);
                    long time2 = System.nanoTime();
                    times.add(time2 - time1);
                    avg += (time2 - time1);
                }

                avg /= NUM_TRIES;
                Collections.sort(times);

                log.info("Shape: {}; Threshold: {}; Average: {} us; p50: {} us", Arrays.toString(shape), threshold, avg / 1000, times.get(times.size() / 2) / 1000);
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
