using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace Socket_client
{
    public class soc
    {
        // Main Variables
        int port; //Port number of server
        string message; //Message to send
        int byteCount; //Raw bytes to send
        NetworkStream stream; //Link stream
        byte[] sendData; //Raw data to send
        byte[] receiveData; //Raw data to receive
        TcpClient client; //TCP client variable
        int f = 0;
        String host;
        public soc(int port, string message, string host)
        {
            this.port = port;
            this.message = message;
            this.host = host;
        }


        //toolStripButton1 -- Open Connection
        public void con()
        {
            try
            {
                client = new TcpClient(host, port);
                Console.WriteLine("connection made");
                //Adds debug to list box and shows message box
            }
            catch (System.Net.Sockets.SocketException)
            {
                Console.WriteLine("Connection Failed");
            }
        }


        private void close()
        {
            client.Close(); //Closes socket
            Console.WriteLine("Connection terminated"); //Adds debug message to list box
        }

        private string receive(int bytecount, TcpClient client, byte[] msg)
        {
            client.ReceiveBufferSize = bytecount;
            if (client.GetStream() != null)
            {
                int read = client.GetStream().Read(msg, 0, msg.Length);
                string msgst = Encoding.ASCII.GetString(msg, 0, read);
                f = 0;
                return msgst;

            }
            else
            {
                f = 1;
                return "no message received";
            }

        }

        static public void Main(String[] args)
        {
            soc s = new soc(65434, "bla bla", "127.0.0.1");
            s.con();
            s.receiveData = new byte[50];
            s.byteCount = s.receiveData.Length;
            while (s.f == 0)
            {
                Console.WriteLine(s.receive(s.byteCount, s.client, s.receiveData));
            }


            s.close();
            Console.WriteLine("Main Method");

        }
    }
}
