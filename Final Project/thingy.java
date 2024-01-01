import java.io.*;

public class thingy {
    public static void main(String[] args) throws IOException {
        String infile = args[0];
        String outfile1 = args[1];
        String outfile2 = args[2];

        int minXcom = 84;
        int minYcom = 84;
        FileInputStream in;
        in = new FileInputStream(infile);
        int[][] thingy = new int[168][168];
        for (int i = 0; i < 168; i++) for (int j = 0; j < 168; j++) {
            int aah = (int) in.read();
            thingy[i][j] = aah;
        }
        PelArray guy = new PelArray(thingy);
        PelArray finalGuy = guy.forceMin(25, 0x00000000);

        int thingy2[][] = new int[minXcom * 2][minYcom * 2];
        thingy2 = finalGuy.getPelArray();

        DataOutputStream out1 = new DataOutputStream(new FileOutputStream(outfile1));
        for (int i = 0; i < 2 * minYcom ; i++) for (int j = 0; j < 2 * minXcom; j++) {
            out1.write((byte) thingy2[i][j]);
        }

        Writer out2 = new FileWriter(outfile2);
        for (int i = 0; i < 2 * minYcom ; i++) for (int j = 0; j < 2 * minXcom; j++) {
            out2.write((String.valueOf(thingy2[i][j] / 255.0) + "\n"));
        }
        in.close();
        out1.close();
        out2.close();
    }
}
