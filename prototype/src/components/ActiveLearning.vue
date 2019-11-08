<template>
  <div id="main">
    <p>Image ID: {{ activationMap.image_id }}</p>
    <p>Disease: {{ activationMap.disease }}</p>
    <div id="container">
      <h2>Canvas Line {{activationMap.canvas_width}} by {{activationMap.canvas_height}}</h2>
      <!--canvas for brain image-->
      <canvas class="canvas" ref="img" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
      <!--canvas for activation map-->
      <canvas class="canvas" ref="map" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
      <!--canvas for corrections-->
      <canvas
        class="canvas"
        ref="draw"
        v-bind:width="canvas_width"
        v-bind:height="canvas_height"
        v-on:mousemove="correctActivation"
      ></canvas>
      <!-- <canvas ref="result" width="436" height="364" style="border:1px solid #000000;"></canvas> -->
    </div>
    <div id="pixels">
      <h2>Pixel Manipulation</h2>
      <canvas
        class="canvas1"
        ref="pixel_img"
        width="436"
        height="364"
        style="border:1px solid #000000;"
        v-on:mousemove="correctTest"
      ></canvas>
    </div>
    <div id="buttons">
      <button v-on:click="setTool('paintbrush')">Paintbrush</button>
      <button v-on:click="setToolSize(5)">5</button>
      <button v-on:click="setToolSize(10)">10</button>
      <button v-on:click="setToolSize(20)">20</button>
      <button v-on:click="setTool('eraser')">Eraser</button>
      <button v-on:click="updateActivationMap">Update Activation Map</button>
      <button v-on:click="onSubmit">Save</button>
    </div>
    <canvas ref="test" width="436" height="364"></canvas>
  </div>
</template>>

<script>
import axios from "axios";
import Header from "./Header";
import Footer from "./Footer";

export default {
  data() {
    return {
      activationMap: {},
      tool_started: false,
      tool: "paintbrush",
      tool_size: 10,
      canvas_width: 436,
      canvas_height: 346
    };
  },
  methods: {
    getActivationMap: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.activationMap = res.data.activation_map;
          this.drawImageAndActivation();
        })
        .catch(error => {
          console.error(error);
        });
    },
    drawImageAndActivation: function() {
      // input: n/a
      // output: load brain image and activation map onto canvas
      let img_ctx = this.$refs.img.getContext("2d");
      // set global opacity
      img_ctx.globalAlpha = 0.8;
      let pixel_ctx = this.$refs.pixel_img.getContext("2d");
      pixel_ctx.globalAlpha = 0.8;
      // draw brain scan
      let img = new Image();
      img.src = require("../assets/brainpic1.jpeg");
      img.onload = () => {
        img_ctx.drawImage(img, 0, 0);
        pixel_ctx.drawImage(img, 0, 0);
      };
      // draw activation map
      // create new ImageData object
      let map_ctx = this.$refs.map.getContext("2d");
      map_ctx.globalAlpha = 0.8;
      let activation_map = map_ctx.createImageData(
        this.canvas_width,
        this.canvas_height
      );
      for (let i = 0; i < activation_map.height; i++) {
        for (let k = 0; k < activation_map.width; k++) {
          let map_index = 4 * (activation_map.width * i + k);
          if (this.activationMap.activation[i][k] === 1) {
            // (255, 0, 0) = red
            activation_map.data[map_index] = 255;
            activation_map.data[map_index + 1] = 0;
            activation_map.data[map_index + 2] = 0;
            // opacity: 0 - 255
            activation_map.data[map_index + 3] = 100;
          } else {
            // (0, 0, 0) = black
            activation_map.data[map_index] = 0;
            activation_map.data[map_index + 1] = 0;
            activation_map.data[map_index + 2] = 0;
            activation_map.data[map_index + 3] = 0;
          }
        }
      }
      map_ctx.putImageData(activation_map, 0, 0);
    },
    correctActivation: function(event) {
      // highlight incorrect activation
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      if (this.tool === "paintbrush") {
        ctx.strokeStyle = "blue";
        ctx.globalCompositeOperation = "source-over";
      } else {
        ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
        ctx.globalCompositeOperation = "destination-out";
      }
      ctx.lineJoin = "round";
      ctx.lineWidth = this.tool_size;
      canvas.onmousedown = () => {
        // start drawing
        this.tool_started = true;
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
      };
      if (this.tool_started) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
      }
      canvas.onmouseup = () => {
        // stop drawing
        if (this.tool_started) {
          this.tool_started = false;
        }
      };
    },
    correctTest: function(event) {
      let canvas = this.$refs.pixel_img;
      let ctx = this.$refs.pixel_img.getContext("2d");
      let map = ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      canvas.onmousedown = () => {
        this.tool_started = true;
      };
      if (this.tool_started) {
        let pixelIndices = this.computeToolPixels(
          event.offsetX,
          event.offsetY,
          this.tool_size
        );
        for (let i = 0; i < pixelIndices.length; i++) {
          let index = (436 * pixelIndices[i][1] + pixelIndices[i][0]) * 4;
          map.data[index] = 0;
          map.data[index + 1] = 0;
          map.data[index + 2] = 0;
          map.data[index + 3] = 255;
        }
        // // single pixel
        // let index = (436 * event.offsetY + event.offsetX) * 4;
        // map.data[index] = 0;
        // map.data[index + 1] = 0;
        // map.data[index + 2] = 0;
        // map.data[index + 3] = 255;
        ctx.putImageData(map, 0, 0);
      }
      canvas.onmouseup = () => {
        if (this.tool_started) {
          this.tool_started = false;
        }
      };
    },
    computeToolPixels: function(x, y, radius) {
      // compute x and y coordinates of all pixels covered by circle with center at (x, y) and radius
      let pixels = [];
      let i = 0;
      let j = 0;
      let i_start = x - radius;
      let j_start = y - radius;
      let i_end = x + radius;
      let j_end = y + radius;
      for (i = i_start; i < i_end; i++) {
        for (j = j_start; j < j_end; j++) {
          if (Math.sqrt((i - x) ** 2 + (j - y) ** 2) < radius) {
            pixels.push([i, j]);
          }
        }
      }
      return pixels;
    },
    setTool: function(tool) {
      this.tool = tool;
    },
    setToolSize: function(tool_size) {
      this.tool_size = tool_size;
    },
    updateActivationMap: function() {
      let map_ctx = this.$refs.map.getContext("2d");
      let draw_ctx = this.$refs.draw.getContext("2d");
      let map_data = map_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      let correction_data = draw_ctx.getImageData(
        0,
        0,
        this.canvas_width,
        this.canvas_height
      );
      let updated_map_data = map_ctx.createImageData(map_data);
      console.log(correction_data);
      for (let i = 0; i < correction_data.data.length; i += 4) {
        if (correction_data.data[i + 2] === 255) {
          map_data.data[i] = 0;
          map_data.data[i + 1] = 0;
          map_data.data[i + 2] = 0;
          map_data.data[i + 3] = 0;
        }
      }
      // clear drawing
      let clear_data = map_ctx.createImageData(map_data);
      draw_ctx.putImageData(clear_data, 0, 0);
      // draw updated activation map
      map_ctx.putImageData(map_data, 0, 0);
    },
    saveData: function(payload) {
      const path = "http://localhost:5000/active_learning";
      axios
        .post(path, payload)
        .then(() => {
          alert("Progress saved!");
          this.getActivationMap();
        })
        .catch(error => {
          console.log(error);
          this.getActivationMap();
        });
    },
    onSubmit: function(event) {
      this.updateActivationMap();
      let map_ctx = this.$refs.map.getContext("2d");
      let corrected_map = map_ctx.getImageData(
        0,
        0,
        this.activationMap.canvas_width,
        this.activationMap.canvas_height
      );
      let test_ctx = this.$refs.test.getContext("2d");
      test_ctx.putImageData(corrected_map, 0, 0);
      const payload = {corrected_activation: corrected_map.data};
      this.saveData(payload);
    }
  },
  components: {
    "app-header": Header,
    "app-footer": Footer
  },
  created() {
    this.getActivationMap();
  },
  mounted() {}
};
</script>

<style scoped>
#main {
  position: relative;
}
#container {
  position: relative;
  float: left;
}
.canvas {
  position: absolute;
  top: 20;
  left: 10;
}
#pixels {
  position: relative;
  float: right;
}
.canvas1 {
  position: relative;
  float: right;
}
#buttons {
  position: relative;
  clear: both;
}
</style>