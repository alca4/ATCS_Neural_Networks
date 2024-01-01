/**
**
** BGR2BMP.java
**
** @author Eric R. Nelson
** September 19, 200.
**
** This code reads in a binary file of byte values that represents either a gray scale (1 byte) or BGR color (3 bytes) images and converts them into a BMP fil**e.
**
** Search for $ to change the data type read in by the program.
**
** Usage:
** java BGR2BMP color/gray width height input_byte_file output_BMP_file
**
** If five arguments are not passed, then a usage message is presented to the user.
**
**/
import java.io.*;

public class BGR2BMP
   {
/*
* Methods to go between little and big endian integer formats since BMP files are little-endian Intel standard (Low-High) while Java is big-endian Motorola (High-Low)
*/
   public int swapInt(int v)
      {
      return ((v >>> 24) | (v << 24) | ((v << 8) & 0x00FF0000) | ((v >> 8) & 0x0000FF00));
      }

   public int swapShort(int v)
      {
      return (((v << 8) & 0xFF00) | ((v >> 8) & 0x00FF));
      }

/*
 *
 * ---- MAIN ----
 *
 */
   public static void main(String[] args)
      {
      int pelCount = 0;
      boolean colorImage, EOF;
      byte byteVal;
      int width, height;
      String inFileName, outFileName;
/*
** Bytes that represent a single 32-bit pel
*/
      byte rgbQuad_rgbBlue;
      byte rgbQuad_rgbGreen;
      byte rgbQuad_rgbRed;
      byte rgbQuad_rgbReserved = 0;
/*
**  BITMAPFILEHEADER
*/
      int bmpFileHeader_bfType;          // WORD
      int bmpFileHeader_bfSize;          // DWORD
      int bmpFileHeader_bfReserved1;     // WORD
      int bmpFileHeader_bfReserved2;     // WORD
      int bmpFileHeader_bfOffBits;       // DWORD
/*
** BITMAPINFOHEADER
*/
      int bmpInfoHeader_biSize;          // DWORD
      int bmpInfoHeader_biWidth;         // LONG
      int bmpInfoHeader_biHeight;        // LONG
      int bmpInfoHeader_biPlanes;        // WORD
      int bmpInfoHeader_biBitCount;      // WORD
      int bmpInfoHeader_biCompression;   // DWORD
      int bmpInfoHeader_biSizeImage;     // DWORD
      int bmpInfoHeader_biXPelsPerMeter; // LONG
      int bmpInfoHeader_biYPelsPerMeter; // LONG
      int bmpInfoHeader_biClrUsed;       // DWORD
      int bmpInfoHeader_biClrImportant;  // DWORD

      BGR2BMP dibdumper = new BGR2BMP(); // needed to get to the byte swapping methods

/*
** args[0] - color or gray, default is gray
** args[1] - image width (columns or x) default is 101
** args[2] - image height (rows or y) default is 101
** args[3] - input activation file default is test1.act
** args[4] - output 32-bit BMP file default is test1.bmp
**
** The default size for the student photos in IC is 175x232 (width, height)
**
*/

      if (args.length != 5)
         System.out.printf("Usage: java BGR2BMP gray/color width height input_byte_file output_BMP_file\n\n");
      else
         {
         colorImage  = args[0].equalsIgnoreCase("color");
         width       = Integer.parseInt(args[1]);
         height      = Integer.parseInt(args[2]);
         inFileName  = args[3];
         outFileName = args[4];

         System.out.printf("Reading ");
         if (colorImage) System.out.printf("BGR color "); else System.out.printf("gray scale ");

         System.out.printf("file '%s' (width = %d, height = %d) and writing out '%s'.\n", inFileName, width, height, outFileName);

/*
** BITMAPFILEHEADER
**
** bfType - Specifies the file type. It must be set to the signature word BM (0x4D42) to indicate bitmap.
** bfSize - Specifies the size, in bytes, of the bitmap file.
** bfReserved1 - Reserved; set to zero
** bfReserved2 - Reserved; set to zero
** bfOffBits - Specifies the offset, in bytes, from the BITMAPFILEHEADER structure to the bitmap bits
*/
         bmpFileHeader_bfType      = 0x424D;                                       // WORD - un-swapped it is the letters BM (0x4D42), but this is Java
         bmpFileHeader_bfOffBits   = 54;                                           // DWORD determined by reading files created by DibDumop
         bmpFileHeader_bfSize      = width * height * 4 + bmpFileHeader_bfOffBits; // DWORD - 4 bytes per pel
         bmpFileHeader_bfReserved1 = 0;                                            // WORD
         bmpFileHeader_bfReserved2 = 0;                                            // WORD

/*
** BITMAPINFOHEADER
**
** biSize        - Specifies the size of the structure, in bytes. This size does not include the color table or the masks mentioned in the biClrUsed member.
**                 See the Remarks section for more information.
** biWidth       - Specifies the width of the bitmap, in pels.
** biHeight      - Specifies the height of the bitmap, in pels.
**                 If biHeight is positive, the bitmap is a bottom-up DIB and its origin is the lower left corner.
**                 If biHeight is negative, the bitmap is a top-down DIB and its origin is the upper left corner.
**                 If biHeight is negative, indicating a top-down DIB, biCompression must be either BI_RGB or BI_BITFIELDS. Top-down DIBs cannot be compressed.
** biPlanes      - Specifies the number of planes for the target device. This value must be set to 1.
** biBitCount    - Specifies the number of bits per pixel (32 in this cse)
** biCompression - Specifies the type of compression for a compressed bottom-up bitmap (top-down DIBs cannot be compressed). It is 0 in our case.
** biSizeImage   -  Specifies the size, in bytes, of the image. This value will be the number of bytes in each scan line which must be padded to
**                  ensure the line is a multiple of 4 bytes (it must align on a DWORD boundary) times the number of rows.
**                  This value may be set to zero for BI_RGB bitmaps (so you cannot be sure it will be set).
** biXPelsPerMeter - Specifies the horizontal resolution, in pixels per meter, of the target device for the bitmap.
** biYPelsPerMeter - Specifies the vertical resolution, in pixels per meter, of the target device for the bitmap
** biClrUsed       - Specifies the number of color indexes in the color table that are actually used by the bitmap.
**                   If this value is zero, the bitmap uses the maximum number of colors corresponding to the value of the biBitCount member for the compression mode specified by biCompression.
** biClrImportant  - Specifies the number of color indexes required for displaying the bitmap. If this value is zero, all colors are required.
*/
         bmpInfoHeader_biSize          = 40;                 // DWORD
         bmpInfoHeader_biWidth         = width;              // LONG
         bmpInfoHeader_biHeight        = height;             // LONG
         bmpInfoHeader_biPlanes        = 1;                  // WORD
         bmpInfoHeader_biBitCount      = 32;                 // WORD
         bmpInfoHeader_biCompression   = 0;                  // DWORD
         bmpInfoHeader_biSizeImage     = width * height * 4; // DWORD We are going to write out a 32-bit image
         bmpInfoHeader_biXPelsPerMeter = 0;                  // LONG
         bmpInfoHeader_biYPelsPerMeter = 0;                  // LONG
         bmpInfoHeader_biClrUsed       = 0;                  // DWORD creating a true color image
         bmpInfoHeader_biClrImportant  = 0;                  // DWORD creating a true color image

         try // lots of things can go wrong when doing file i/o
            {
            FileInputStream fInStream = new FileInputStream(inFileName); // Open the file that is the second command line parameter
            DataInputStream in = new DataInputStream(fInStream);         // Convert our input stream to a DataInputStream

            try
               {
               FileOutputStream fOutStream = new FileOutputStream(outFileName);
               DataOutputStream out = new DataOutputStream(fOutStream);
/*
** BITMAPFILEHEADER
*/
               out.writeShort(bmpFileHeader_bfType);                           // WORD - un-swapped since should be the letters BM
               out.writeInt(dibdumper.swapInt(bmpFileHeader_bfSize));          // DWORD
               out.writeShort(dibdumper.swapShort(bmpFileHeader_bfReserved1)); // WORD
               out.writeShort(dibdumper.swapShort(bmpFileHeader_bfReserved2)); // WORD
               out.writeInt(dibdumper.swapInt(bmpFileHeader_bfOffBits));       // DWORD

/*
** BITMAPINFOHEADER
*/
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biSize));          // DWORD
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biWidth));         // LONG
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biHeight));        // LONG
               out.writeShort(dibdumper.swapShort(bmpInfoHeader_biPlanes));    // WORD
               out.writeShort(dibdumper.swapShort(bmpInfoHeader_biBitCount));  // WORD
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biCompression));   // DWORD
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biSizeImage));     // DWORD
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biXPelsPerMeter)); // LONG
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biYPelsPerMeter)); // LONG
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biClrUsed));       // DWORD
               out.writeInt(dibdumper.swapInt(bmpInfoHeader_biClrImportant));  // DWORD

/*
** Read in 1 or 3 byte values and then write them our as an RGB_Quad value for a 32-bit BMP image which are in the order Blue-Green-Red-Zero.
*/
               EOF = false;
               while (!EOF)
                  {
                  try // an exception is thrown when end of file is reached
                     {
                     if (colorImage)
                        {
                        rgbQuad_rgbBlue  = in.readByte();
                        rgbQuad_rgbGreen = in.readByte();
                        rgbQuad_rgbRed   = in.readByte();
                        }
                     else
                        {
                        byteVal = in.readByte();    // $$$
                        rgbQuad_rgbBlue  = byteVal;
                        rgbQuad_rgbGreen = byteVal;
                        rgbQuad_rgbRed   = byteVal;
                        }

                     out.writeByte(rgbQuad_rgbBlue);
                     out.writeByte(rgbQuad_rgbGreen);
                     out.writeByte(rgbQuad_rgbRed);
                     out.writeByte(rgbQuad_rgbReserved); // The reserved value is zero

                     ++pelCount;
                     } // try
                  catch (Exception e)
                     {
                     EOF = true;

                     if (pelCount != width * height)
                        {
                        System.out.printf("%d pels read out of %d expected\n", pelCount, width * height);
                        System.err.println("Reason for stopping: " + e);
                        }
                     }
                  } // while (!EOF)

               out.close();
               fOutStream.close();
               } // try
            catch (Exception e)
               {
               System.err.println("File output error" + e);
               }

            in.close();
            fInStream.close();
            } // try
         catch (Exception e)
            {
            System.err.println("File input error" + e);
            }
         } // if (args.length != 5) ... else
      } // public public void main
   } // public class DibDump
