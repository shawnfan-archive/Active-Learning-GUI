<template>
  <div>
    <app-header></app-header>
    <p>Subject ID: {{ patients.subject_id }}</p>
    <p>Subject group: {{ patients.disease }}</p>
    <p>Total number of views: {{ patients.number_of_views }}</p>
    <div id="canvas" v-on:click="collectXY">{{x}}, {{y}}</div>
    <button v-on:click="onSubmit">Save</button>
    <app-footer></app-footer>
  </div>
</template>>

<script>
import axios from "axios";
import Header from "./Header";
import Footer from "./Footer";

export default {
  data() {
    return {
      patients: {},
      x: [],
      y: []
    };
  },
  methods: {
    getPatients: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.patients = res.data.patients;
        })
        .catch(error => {
          console.error(error);
        });
    },
    collectXY: function(event) {
      this.x.push(event.offsetX);
      this.y.push(event.offsetY);
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          alert('Progress saved!');
          this.getPatients();
        })
        .catch(error => {
          console.log(error);
          this.getPatients();
        });
    },
    onSubmit: function(event) {
      const payload = {
        x_coord: this.x,
        y_coord: this.y
      };
      this.saveData(payload);
    }
  },
  components: {
    "app-header": Header,
    "app-footer": Footer
  },
  created() {
    alert("created");
    this.getPatients();
  }
};
</script>

<style scoped>
#canvas {
  width: 600px;
  padding: 200px 20px;
  text-align: center;
  border: 1px solid #333;
}
</style>