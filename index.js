import { Table, Typography, Button, Select, Modal, message } from "antd";
import axios from 'axios';
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import './tampil.css';

const { Option } = Select;
const { Title } = Typography;

// Fungsi untuk mendapatkan data gabungan berdasarkan tahun
export const getCombinedData = (year) => {
  const url = year ? `http://localhost:8081/get_combined_data?year=${year}` : 'http://localhost:8081/get_combined_data';
  return fetch(url)
      .then(response => response.json());
}

function DeleteCombinedFile({ onSuccess, selectedYear, isLoading, setIsLoading }) {
  const showDeleteConfirm = () => {
    Modal.confirm({
      title: `Apakah Anda yakin ingin menghapus data tahun ${selectedYear}?`,
      content: 'Tindakan ini tidak dapat dibatalkan',
      okText: 'Ya, hapus',
      okType: 'danger',
      cancelText: 'Batal',
      onOk: handleDelete,
    });
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      const response = await axios.delete(`http://localhost:8081/delete_data_by_upload_year?upload_year=${selectedYear}`);
      message.success(response.data.message);
      onSuccess();
    } catch (error) {
      message.error(error.response ? error.response.data.detail : 'Error deleting data');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Button 
      type="primary" 
      onClick={showDeleteConfirm} 
      loading={isLoading}
      style={{ marginLeft: '8px', fontSize:'13px'}} 
    >
      Hapus Data Tahun {selectedYear}
    </Button>
  );
}

const columns = [
  {
    title: 'Name of Vendor',
    dataIndex: 'nof',
    key: 'nof',
  },
  {
    title: 'Short Text',
    dataIndex: 'st',
    key: 'st',
  },
  {
    title: 'PO Date',
    dataIndex: 'po',
    key: 'po',
  },
  {
    title: 'Delivery Date',
    dataIndex: 'dd',
    key: 'dd',
  },
  {
    title: 'Doc Date GR',
    dataIndex: 'ddgr',
    key: 'ddgr',
  },
  {
    title: 'Local Amount',
    dataIndex: 'la',
    key: 'la',
  },
  {
    title: 'Jumlah Projek',
    dataIndex: 'jp',
    key: 'jp',
  },
  {
    title: 'Nilai Performansi KHS',
    dataIndex: 'khs',
    key: 'khs',
  },
  {
    title: 'Alker/Salker',
    dataIndex: 'alker',
    key: 'alker',
  },
  {
    title: 'Stok Material',
    dataIndex: 'stok',
    key: 'stok',
  },
  {
    title: 'Jumlah Team',
    dataIndex: 'tim',
    key: 'tim',
  },
  {
    title: 'Kerapihan',
    dataIndex: 'rapih',
    key: 'rapih',
  },
];


function Tampil() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState([]);
  const [error, setError] = useState(null);
  const [tahunTerpilih, setTahunTerpilih] = useState();
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [isLoading, setIsLoading] = useState(false);
  
  const tahun = [...Array(new Date().getFullYear() - 2019).keys()].map(x => x + 2020);

  const handleGenerateKlasifikasi = () => {
    setLoading(true);
    fetch("http://localhost:8081/run_klasifikasi", { method: "POST" })
      .then((response) => response.json())
      .then((data) => {
        // Handle response data jika diperlukan
        setLoading(false);
        // Refresh data setelah klasifikasi selesai
        refreshData();
        // Arahkan pengguna ke halaman prediksi
        navigate('/prediksi'); 
      })
      .catch((err) => {
        console.error("API error:", err);
        setError(
          "Failed to run classification. Please check your connection and try again."
        );
        setLoading(false);
      });
  };

  // Fungsi handler baru untuk perubahan tahun
  const handleYearChange = (tahun) => {
    setTahunTerpilih(tahun);
    setSelectedYear(tahun);
    refreshData(tahun); 
  };

    const refreshData = (tahun) => {
      setLoading(true);
      getCombinedData(tahun)
        .then((data) => {  
          if (Array.isArray(data)) {
              setDataSource(data);
            } else {
              console.warn("Unexpected data structure from API");
              setError("Failed to load data. Please try again later.");
            }
            setLoading(false);
          })
          .catch((err) => {
            console.error("API error:", err);
            setError(
              "Failed to load data. Please check your connection and try again."
            );
            setLoading(false);
          });
      };


    useEffect(() => {
        // Panggil fungsi untuk mengambil data saat komponen dimuat dengan tahun tertentu
        refreshData(tahunTerpilih); // Atau Anda bisa mengatur default di sini jika diperlukan
    }, [tahunTerpilih]);
    

  return (
    <div>
      <div className="tampil">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '50px', marginLeft: '50px', marginTop: '15px' }}>
          {/* Dropdown untuk pemilihan tahun */}
          <Select defaultValue="Pilih Tahun" style={{ width: 120 }} onChange={handleYearChange}>
            {tahun.map(tahun => (
              <Option key={tahun} value={tahun}>{tahun}</Option>
            ))}
          </Select>
          <Button type="primary" onClick={handleGenerateKlasifikasi} loading={loading}>
            Generate Klasifikasi
          </Button>
        </div>
        <div className="delete-button-wrapper" style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end', marginRight: '810px', marginBottom: '20px'}}>
          <DeleteCombinedFile
            onSuccess={() => refreshData(tahunTerpilih)}
            selectedYear={selectedYear}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>
        {error && <div style={{ color: 'red', marginBottom: '16px' }}>{error}</div>}
        <div className="table-container">
          <Table style={{ minWidth: '1500px' }} columns={columns} dataSource={dataSource} loading={loading} rowKey="id" />
        </div>
      </div>
    </div>
  );
}

export default Tampil;