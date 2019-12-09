<template>
  <div id="main">
    <h1>Image ID: {{ image.image_id }} Disease: {{ image.disease }}</h1>
    <div id="container">
      <h2>Canvas Line {{canvas_width}} by {{canvas_height}}</h2>
      <!--canvas for brain image-->
      <div class="canvas">
        <canvas ref="img" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
      </div>
      <!--canvas for activation map-->
      <div class="canvas">
        <canvas ref="map" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
      </div>
      <div class="canvas">
        <canvas ref="cursor" v-bind:width="canvas_width" v-bind:height="canvas_height"></canvas>
      </div>
      <!--canvas for corrections-->
      <div id="draw">
        <canvas
          ref="draw"
          v-bind:width="canvas_width"
          v-bind:height="canvas_height"
          v-on:mousemove="correctActivation"
        ></canvas>
      </div>
    </div>
    <div id="load">
      <button id="load_button" v-on:click="loadActivationMap">Load Activation Map</button>
    </div>
    <div id="save">
      <button id="save_button" v-on:click="onSubmit">Save and Retrain Model</button>
    </div>
    <div id="graphics">
      <div id="mode_buttons">
        <h3>Mode:</h3>
        <div>
          <button class="mode_button" v-on:click="setTool('activate')">Activate</button>
        </div>
        <div>
          <button class="mode_button" v-on:click="setTool('deactivate')">Deactivate</button>
        </div>
      </div>
      <div id="size_buttons">
        <h3>Paintbrush Size:</h3>
        <button class="size_button" v-on:click="setToolSize(5)">5</button>
        <button class="size_button" v-on:click="setToolSize(10)">10</button>
        <button class="size_button" v-on:click="setToolSize(20)">20</button>
      </div>
    </div>
    <!-- <canvas ref="test" width="436" height="364"></canvas> -->
  </div>
</template>>

<script>
import axios from "axios";

export default {
  data() {
    return {
      image: {},
      activationMap: {},
      tool: "deactivate",
      tool_started: false,
      tool_size: 10,
      canvas_width: 436,
      canvas_height: 364,
      graphics: {
        // rgba
        activation_color: [255, 0, 0, 100],
        inactivation_color: [0, 0, 0, 0],
        paintbrush: "rgba(0, 0, 255, 255)",
        image_opacity: 1.0,
        map_opacity: 0.8
      }
    };
  },
  methods: {
    setTool: function(tool) {
      this.tool = tool;
    },
    setToolSize: function(tool_size) {
      this.tool_size = tool_size;
    },
    getActivationMap: function() {
      const path = "http://localhost:5000/active_learning";
      axios
        .get(path)
        .then(res => {
          this.image = res.data.image;
          this.activationMap = res.data.activation_map;
          this.loadImage();
        })
        .catch(error => {
          console.error(error);
        });
    },
    loadImage: function() {
      let img_ctx = this.$refs.img.getContext("2d");
      // set global opacity
      img_ctx.globalAlpha = this.graphics.image_opacity;
      // load brain image
      let img = new Image();
      //img.src = require("/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/prototype/src/assets/brainpic1.jpeg")
      img.src = require("/Users/shawnfan/Dropbox/active_learning_20191115/Active-Learning-GUI/flask_unet/output/resized_input.png");
      img.onload = () => {
        img_ctx.drawImage(img, 0, 0);
      };
    },
    loadCursor: function(x, y) {
      let ctx = this.$refs.cursor.getContext("2d");
      ctx.clearRect(0, 0, this.canvas_width, this.canvas_height);
      ctx.fillStyle = "rgba(255,255,255,255)";
      ctx.beginPath();
      ctx.arc(x, y, this.tool_size / 2, 0, 2 * Math.PI);
      ctx.fill();
    },
    loadActivationMap: function() {
      let map_ctx = this.$refs.map.getContext("2d");
      map_ctx.globalAlpha = this.graphics.map_opacity;
      let map_data = map_ctx.createImageData(
        this.canvas_width,
        this.canvas_height
      );
      for (let i = 0; i < map_data.height; i++) {
        for (let k = 0; k < map_data.width; k++) {
          let map_index = 4 * (map_data.width * i + k);
          if (this.activationMap.activation[i][k] === 1) {
            map_data.data[map_index] = this.graphics.activation_color[0];
            map_data.data[map_index + 1] = this.graphics.activation_color[1];
            map_data.data[map_index + 2] = this.graphics.activation_color[2];
            map_data.data[map_index + 3] = this.graphics.activation_color[3];
          } else {
            map_data.data[map_index] = this.graphics.inactivation_color[0];
            map_data.data[map_index + 1] = this.graphics.inactivation_color[1];
            map_data.data[map_index + 2] = this.graphics.inactivation_color[2];
            map_data.data[map_index + 3] = this.graphics.inactivation_color[3];
          }
        }
      }
      map_ctx.putImageData(map_data, 0, 0);
    },
    correctActivation: function(event) {
      // load cursor
      this.loadCursor(event.offsetX, event.offsetY);

      // highlight incorrect activation
      let canvas = this.$refs.draw;
      let ctx = this.$refs.draw.getContext("2d");
      ctx.strokeStyle = this.graphics.paintbrush;
      ctx.fillStyle = this.graphics.paintbrush;
      ctx.globalCompositeOperation = "source-over";
      ctx.lineJoin = "round";
      ctx.lineWidth = this.tool_size;
      // start drawing
      canvas.onmousedown = () => {
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
        this.tool_started = true;
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
      };
      if (this.tool_started) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
        this.updateActivationMap();
      }
      canvas.onmouseup = () => {
        // stop drawing
        if (this.tool_started) {
          this.tool_started = false;
        }
        // draw circle
        ctx.beginPath();
        ctx.arc(
          event.offsetX,
          event.offsetY,
          this.tool_size / 2,
          0,
          2 * Math.PI
        );
        ctx.fill();
        this.updateActivationMap();
      };
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

      let pixel_color = this.graphics.activation_color;
      if (this.tool === "deactivate") {
        pixel_color = this.graphics.inactivation_color;
      }

      let updated_map_data = map_ctx.createImageData(map_data);
      for (let i = 0; i < correction_data.data.length; i += 4) {
        if (correction_data.data[i + 2] === 255) {
          map_data.data[i] = pixel_color[0];
          map_data.data[i + 1] = pixel_color[1];
          map_data.data[i + 2] = pixel_color[2];
          map_data.data[i + 3] = pixel_color[3];
        }
      }

      // draw updated activation map
      map_ctx.putImageData(map_data, 0, 0);

      // clear draw
      let clear_data = map_ctx.createImageData(map_data);
      draw_ctx.putImageData(clear_data, 0, 0);
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
        this.canvas_width,
        this.canvas_height
      );
      let test_ctx = this.$refs.test.getContext("2d");
      test_ctx.putImageData(corrected_map, 0, 0);
      const payload = { corrected_activation: corrected_map.data };
      this.saveData(payload);
    }
  },
  created() {
    this.getActivationMap();
  },
  mounted() {}
};
</script>

<style scoped>
#container {
  position: relative;
}
.canvas {
  position: absolute;
  cursor: none;
}
#draw {
  position: relative;
  cursor: none;
  float: left;
}
#load {
  position: relative;
  float: left;
  padding: 1em;
}
#load_button {
  background-color: white;
  color: navy;
  text-align: center;
  font-size: 26px;
  font-family: Arial, Helvetica, sans-serif;
}
#save {
  position: relative;
  float: left;
  padding: 1em
}
#save_button {
  background-color: white;
  color: navy;
  text-align: center;
  font-size: 26px;
  font-family: Arial, Helvetica, sans-serif;
}
#graphics {
  position: relative;
  float: left;
  padding: 1em;
}
#mode_buttons{
  float: left;
}
.mode_button {
  font-size: 24px;
  border-radius: 40%;
  padding: 0.5em;
}
#size_buttons {
  float: left;
}
.size_button {
  font-size: 20px;
  border-radius: 50%;
}
</style>