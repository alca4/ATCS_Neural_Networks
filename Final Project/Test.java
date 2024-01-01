/*
 * Eric R. Nelson
 * 3/4/2009
 * 
 * This class is used to test the methods in the PelArray class and the DIBitmap class
 *
 * javac Test.java PelArray.java
 *
 */
public class Test
   {
   public static int RED   = 0x00FF0000;
   public static int GREEN = 0x0000FF00;
   public static int BLUE  = 0x000000FF;

   public static int BLACK  = 0x00000000;
   public static int WHITE  = 0x00FFFFFF;

   public static void main(String[] args)
      {
      int[][] box = {{RED,   RED,   RED,   RED, RED},   // Red border around a green box with a blue interior
                     {RED, GREEN, GREEN, GREEN, RED},
                     {RED, GREEN,  BLUE, GREEN, RED},
                     {RED, GREEN,  BLUE, GREEN, RED},
                     {RED, GREEN,  BLUE, GREEN, RED},
                     {RED, GREEN, GREEN, GREEN, RED},
                     {RED,   RED,   RED,   RED, RED}
                    };

      int[][] arrowL = {{BLACK, BLACK, WHITE, BLACK, BLACK}, // Arrow with an L at the bottom to test rotation
                        {BLACK, WHITE, WHITE, WHITE, BLACK},
                        {WHITE, WHITE, WHITE, WHITE, WHITE},
                        {BLACK, BLACK, WHITE, BLACK, BLACK},
                        {BLACK, BLACK, WHITE, BLACK, BLACK},
                        {BLACK, BLACK, WHITE, BLACK, BLACK},
                        {BLACK, BLACK, WHITE, BLACK, BLACK},
                        {BLACK, BLACK, WHITE, WHITE, WHITE}
                       };
                      
      int[][] corner = {{BLACK, BLACK, BLACK, BLACK, BLACK}, // Lower right corner is white
                        {BLACK, BLACK, BLACK, BLACK, BLACK},
                        {BLACK, BLACK, BLACK, BLACK, BLACK},
                        {BLACK, BLACK, BLACK, BLACK, BLACK},
                        {BLACK, BLACK, BLACK, BLACK, BLACK},
                        {BLACK, BLACK, WHITE, WHITE, WHITE},
                        {BLACK, BLACK, WHITE, WHITE, WHITE},
                        {BLACK, BLACK, WHITE, WHITE, WHITE}
                       };
                      


      PelArray pixels = new PelArray(arrowL);

      System.out.printf("Image Width %d\n", pixels.getWidth());
      System.out.printf("Image Height %d\n\n", pixels.getHeight());

      pixels.dump();
      System.out.println();

      System.out.println("Rotate Counter Clockwise");
      PelArray pixRot = pixels.rotateCCW90();
      pixRot.dump();
      System.out.println();

      System.out.println("Rotate Clockwise");
      pixRot = pixels.rotateCW90();
      pixRot.dump();
      System.out.println();

      System.out.println("Flip Horizontal");
      pixRot = pixels. flipHorizontal();
      pixRot.dump();
      System.out.println();

      System.out.println("Flip Vertical");
      pixRot = pixels. flipVertical();
      pixRot.dump();
      System.out.println();

/*
** The following shows how to take the center of mass and then shift the image to make that point the geometric center of the image.
** The shift will crop anything that falls outside the dimensions of the current image and fills new pels with zero.
*/

      System.out.println("Make BLUE Only and center");

      PelArray pixels2 = new PelArray(corner);           // Create a new pelArray from the corner array of bytes
      pixels2.dump();
      System.out.println();

      PelArray pixels3 = pixels2.grayScaleImage();       // Convert the image to gray

      PelArray blueOnly = pixels3.oneColorImage(BLUE);   // Mask all but the least significant byte of the colors
      blueOnly.dump();
      System.out.println();

      System.out.printf("COM = %d, %d\n", blueOnly.getXcom(), blueOnly.getYcom());
      System.out.println();

      PelArray centered = blueOnly.offset(blueOnly.getWidth()/2 - blueOnly.getXcom(), blueOnly.getHeight()/2 - blueOnly.getYcom()); // Move the COM to the image center
      centered.dump();


/* In case you want to play with the box and not the arrow.
      pixels = new PelArray(box);

      pixels.dump();
      System.out.println();

      System.out.printf("\n(X,Y) = (%d, %d)\n\n", pixels.getXcom(), pixels.getYcom());

      PelArray pix10 = pixels.grayScaleImage();
      pix10.dump();
      System.out.println();

      pix10 = pixels.onesComplimentImage();
      pix10.dump();
      System.out.println();

      pix10 = pixels. oneColorImage(RED);
      pix10.dump();
      System.out.println();

      pix10 = pixels. oneColorImage(GREEN);
      pix10.dump();
      System.out.println();

      pix10 = pixels. oneColorImage(BLUE);
      pix10.dump();
      System.out.println();
*/


      }
   }
